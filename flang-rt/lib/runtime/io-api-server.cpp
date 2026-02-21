//===-- lib/runtime/io-api-server.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the RPC server-side handlling of the I/O statement API needed for
// basic list-directed output (PRINT *) of intrinsic types for the GPU.

#include "array.h"
#include "io-api-gpu.h"
#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Runtime/io-api.h"
#include <cstdlib>
#include <cstring>
#include <tuple>

#include <shared/rpc.h>
#include <shared/rpc_dispatch.h>

namespace Fortran::runtime::io {
namespace {

// Context used to chain the IO operations once run.
struct IOContext {
  Cookie cookie = nullptr;
  enum Iostat result = IostatOk;
};

// The base class to store deferred execution of a function. Uses function
// pointers for type erasure to avoid virtual dispatch.
struct DeferredFunctionBase {
  using ExecuteFn = void (*)(void *, IOContext &);
  using DestroyFn = void (*)(void *);

  DeferredFunctionBase(void *impl, ExecuteFn exec, DestroyFn dtor)
      : impl_(impl), execute_(exec), destroy_(dtor) {}

  DeferredFunctionBase(const DeferredFunctionBase &) = delete;
  DeferredFunctionBase &operator=(const DeferredFunctionBase &) = delete;
  DeferredFunctionBase(DeferredFunctionBase &&other)
      : impl_(other.impl_), execute_(other.execute_), destroy_(other.destroy_) {
    other.impl_ = nullptr;
  }
  DeferredFunctionBase &operator=(DeferredFunctionBase &&other) {
    if (this != &other) {
      reset();
      impl_ = other.impl_;
      execute_ = other.execute_;
      destroy_ = other.destroy_;
      other.impl_ = nullptr;
    }
    return *this;
  }

  ~DeferredFunctionBase() { reset(); }

  void execute(IOContext &ctx) { execute_(impl_, ctx); }

  static OwningPtr<char> TempString(const char *str, std::size_t size) {
    if (!str) {
      return {};
    }

    OwningPtr<char> temp = SizedNew<char>{Terminator{__FILE__, __LINE__}}(size);
    std::memcpy(temp.get(), str, size);
    return OwningPtr<char>(temp.release());
  }

  static OwningPtr<char> TempString(const char *str) {
    if (!str) {
      return {};
    }
    return TempString(str, std::strlen(str) + 1);
  }

private:
  void reset() {
    if (impl_) {
      destroy_(impl_);
      FreeMemory(impl_);
      impl_ = nullptr;
    }
  }

  void *impl_ = nullptr;
  ExecuteFn execute_ = nullptr;
  DestroyFn destroy_ = nullptr;
};

// Fortran does not support nested or recursive I/O, which is problematic for
// parallel execution on a GPU. To support this, we defer execution of runtime
// functions coming from the GPU's client until the end of that sequence is
// reached. This allows us to finish them in a single pass.
template <typename FnTy, typename... Args> struct DeferredFunction {
  FnTy fn_;
  std::tuple<std::decay_t<Args>...> args_;

  DeferredFunction(FnTy &&fn, Args &&...args)
      : fn_(std::forward<FnTy>(fn)), args_(std::forward<Args>(args)...) {}

  // When executing the final command queue we need to replace the temporary
  // values obtained from the GPU with the returned values from the actual
  // runtime functions.
  void execute(IOContext &ctx) {
    auto caller = [&](auto &&...args) { return fn_(Rewrite(args, ctx)...); };

    using RetTy = std::invoke_result_t<FnTy,
        decltype(Rewrite(std::declval<Args &>(), ctx))...>;
    if constexpr (std::is_same_v<RetTy, Cookie>) {
      ctx.cookie = std::apply(caller, args_);
    } else if constexpr (std::is_same_v<RetTy, Iostat>) {
      ctx.result = std::apply(caller, args_);
    } else {
      std::apply(caller, args_);
    }
  }

private:
  template <typename T> T &Rewrite(T &v, IOContext &) { return v; }

  const char *Rewrite(OwningPtr<char> &p, IOContext &) { return p.get(); }

  Cookie Rewrite(Cookie, IOContext &ctx) { return ctx.cookie; }
};

template <typename Fn, typename... Args>
DeferredFunctionBase MakeDeferred(Fn &&fn, Args &&...args) {
  Terminator terminator{__FILE__, __LINE__};
  using Ty = DeferredFunction<Fn, Args...>;
  auto ptr = SizedNew<Ty>{terminator}(
      sizeof(Ty), std::forward<Fn>(fn), std::forward<Args>(args)...);
  void *raw = ptr.release();
  return DeferredFunctionBase(
      raw,
      [](void *self, IOContext &ctx) { static_cast<Ty *>(self)->execute(ctx); },
      [](void *self) { static_cast<Ty *>(self)->~Ty(); });
}

// The context associated with the queue of deferred functions. This serves as
// our cookie object while executing this on the GPU.
struct DeferredContext {
  IOContext ioCtx;
  DynamicArray<DeferredFunctionBase> commands;
};

template <typename FnTy, typename... Args>
bool EnqueueDeferred(FnTy &&fn, Cookie cookie, Args &&...args) {
  DeferredContext *ctx = reinterpret_cast<DeferredContext *>(cookie);
  ctx->commands.emplace_back(
      MakeDeferred(fn, cookie, std::forward<Args>(args)...));
  return true;
}

template <std::uint32_t NumLanes>
rpc::Status HandleOpcodesImpl(rpc::Server::Port &port) {
  switch (port.get_opcode()) {
  case BeginExternalListOutput_Opcode:
    rpc::invoke<NumLanes>(port,
        [](ExternalUnit unitNumber, const char *sourceFile,
            int sourceLine) -> Cookie {
          DeferredContext *ctx = new (AllocateMemoryOrCrash(
              Terminator{__FILE__, __LINE__}, sizeof(DeferredContext)))
              DeferredContext;

          ctx->commands.emplace_back(
              MakeDeferred(IONAME(BeginExternalListOutput), unitNumber,
                  DeferredFunctionBase::TempString(sourceFile), sourceLine));

          return reinterpret_cast<Cookie>(ctx);
        });
    break;
  case BeginExternalFormattedOutput_Opcode:
    rpc::invoke<NumLanes>(port,
        [](const char *format, std::size_t formatLength,
            const Descriptor *formatDescriptor, ExternalUnit unitNumber,
            const char *sourceFile, int sourceLine) -> Cookie {
          Terminator terminator{__FILE__, __LINE__};
          if (formatDescriptor)
            terminator.Crash("Non-trivial format descriptors are unsupported");

          DeferredContext *ctx =
              new (AllocateMemoryOrCrash(terminator, sizeof(DeferredContext)))
                  DeferredContext;

          ctx->commands.emplace_back(
              MakeDeferred(IONAME(BeginExternalFormattedOutput),
                  DeferredFunctionBase::TempString(format, formatLength),
                  formatLength, formatDescriptor, unitNumber,
                  DeferredFunctionBase::TempString(sourceFile), sourceLine));

          return reinterpret_cast<Cookie>(ctx);
        });
    break;
  case EnableHandlers_Opcode:
    rpc::invoke<NumLanes>(port,
        [](Cookie cookie, bool hasIoStat, bool hasErr, bool hasEnd, bool hasEor,
            bool hasIoMsg) -> void {
          EnqueueDeferred(IONAME(EnableHandlers), cookie, hasIoStat, hasErr,
              hasEnd, hasEor, hasIoMsg);
        });
    break;
  case EndIoStatement_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie) -> Iostat {
      DeferredContext *ctx = reinterpret_cast<DeferredContext *>(cookie);

      ctx->commands.emplace_back(MakeDeferred(IONAME(EndIoStatement), cookie));
      for (auto &fn : ctx->commands) {
        fn.execute(ctx->ioCtx);
      }
      Iostat result = ctx->ioCtx.result;

      ctx->~DeferredContext();
      FreeMemory(ctx);

      return result;
    });
    break;
  case OutputInteger8_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie, std::int8_t n) -> bool {
      return EnqueueDeferred(IONAME(OutputInteger8), cookie, n);
    });
    break;
  case OutputInteger16_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie, std::int16_t n) -> bool {
      return EnqueueDeferred(IONAME(OutputInteger16), cookie, n);
    });
    break;
  case OutputInteger32_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie, std::int32_t n) -> bool {
      return EnqueueDeferred(IONAME(OutputInteger32), cookie, n);
    });
    break;
  case OutputInteger64_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie, std::int64_t n) -> bool {
      return EnqueueDeferred(IONAME(OutputInteger64), cookie, n);
    });
    break;
#ifdef __SIZEOF_INT128__
  case OutputInteger128_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie, common::int128_t n) -> bool {
      return EnqueueDeferred(IONAME(OutputInteger128), cookie, n);
    });
    break;
#endif
  case OutputReal32_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie, float x) -> bool {
      return EnqueueDeferred(IONAME(OutputReal32), cookie, x);
    });
    break;
  case OutputReal64_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie, double x) -> bool {
      return EnqueueDeferred(IONAME(OutputReal64), cookie, x);
    });
    break;
  case OutputComplex32_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie, float re, float im) -> bool {
      return EnqueueDeferred(IONAME(OutputComplex32), cookie, re, im);
    });
    break;
  case OutputComplex64_Opcode:
    rpc::invoke<NumLanes>(
        port, [](Cookie cookie, double re, double im) -> bool {
          return EnqueueDeferred(IONAME(OutputComplex64), cookie, re, im);
        });
    break;
  case OutputAscii_Opcode:
    rpc::invoke<NumLanes>(
        port, [](Cookie cookie, const char *x, std::size_t length) -> bool {
          return EnqueueDeferred(IONAME(OutputAscii), cookie,
              DeferredFunctionBase::TempString(x, length), length);
        });
    break;
  case OutputCharacter_Opcode:
    rpc::invoke<NumLanes>(port,
        [](Cookie cookie, const char *x, std::size_t length, int kind) -> bool {
          return EnqueueDeferred(IONAME(OutputCharacter), cookie,
              DeferredFunctionBase::TempString(x, length * kind), length, kind);
        });
    break;
  case OutputLogical_Opcode:
    rpc::invoke<NumLanes>(port, [](Cookie cookie, bool truth) -> bool {
      return EnqueueDeferred(IONAME(OutputLogical), cookie, truth);
    });
    break;
  default:
    return rpc::RPC_UNHANDLED_OPCODE;
  }

  return rpc::RPC_SUCCESS;
}
} // namespace

RT_EXT_API_GROUP_BEGIN
std::uint32_t IODEF(HandleRPCOpcodes)(void *raw, std::uint32_t numLanes) {
  rpc::Server::Port &port = *reinterpret_cast<rpc::Server::Port *>(raw);
  switch (numLanes) {
  case 1:
    return HandleOpcodesImpl<1>(port);
  case 32:
    return HandleOpcodesImpl<32>(port);
  case 64:
    return HandleOpcodesImpl<64>(port);
  default:
    return rpc::RPC_ERROR;
  }
}
RT_EXT_API_GROUP_END
} // namespace Fortran::runtime::io
