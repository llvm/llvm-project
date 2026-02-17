//===-- Helper functions for client / server dispatch -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SHARED_RPC_DISPATCH_H
#define LLVM_LIBC_SHARED_RPC_DISPATCH_H

#include "rpc.h"
#include "rpc_util.h"

namespace rpc {
namespace {

// Forward declarations needed for the server, we assume these are present.
extern "C" void *malloc(__SIZE_TYPE__);
extern "C" void free(void *);

// Traits to convert between a tuple and binary representation of an argument
// list.
template <typename... Ts> struct tuple_bytes {
  static constexpr uint64_t SIZE = rpc::max(1ul, (0 + ... + sizeof(Ts)));
  using array_type = rpc::array<uint8_t, SIZE>;

  template <uint64_t... Is>
  RPC_ATTRS static constexpr array_type pack_impl(rpc::tuple<Ts...> t,
                                                  rpc::index_sequence<Is...>) {
    array_type out{};
    uint8_t *p = out.data();
    ((rpc::rpc_memcpy(p, &rpc::get<Is>(t), sizeof(Ts)), p += sizeof(Ts)), ...);
    return out;
  }

  RPC_ATTRS static constexpr array_type pack(rpc::tuple<Ts...> t) {
    return pack_impl(t, rpc::index_sequence_for<Ts...>{});
  }

  template <uint64_t... Is>
  RPC_ATTRS static constexpr rpc::tuple<Ts...>
  unpack_impl(const uint8_t *data, rpc::index_sequence<Is...>) {
    rpc::tuple<Ts...> t{};
    const uint8_t *p = data;
    ((rpc::rpc_memcpy(&rpc::get<Is>(t), p, sizeof(Ts)), p += sizeof(Ts)), ...);
    return t;
  }

  RPC_ATTRS static constexpr rpc::tuple<Ts...> unpack(const array_type &a) {
    return unpack_impl(a.data(), rpc::index_sequence_for<Ts...>{});
  }
};
template <typename... Ts>
struct tuple_bytes<rpc::tuple<Ts...>> : tuple_bytes<Ts...> {};

// Client-side dispatch of pointer values. We copy the memory associated with
// the pointer to the server and receive back a server-side pointer to replace
// the client-side pointer in the argument list.
template <uint64_t Idx, typename Tuple>
RPC_ATTRS constexpr void prepare_arg(rpc::Client::Port &port, Tuple &t) {
  using ArgTy = rpc::tuple_element_t<Idx, Tuple>;
  using ElemTy = rpc::remove_pointer_t<ArgTy>;
  if constexpr (rpc::is_pointer_v<ArgTy> && rpc::is_complete_v<ElemTy> &&
                !rpc::is_void_v<ElemTy>) {
    // We assume all constant character arrays are C-strings.
    uint64_t size{};
    if constexpr (rpc::is_same_v<ArgTy, const char *>)
      size = rpc::string_length(rpc::get<Idx>(t));
    else
      size = sizeof(rpc::remove_pointer_t<ArgTy>);
    port.send_n(rpc::get<Idx>(t), size);
    port.recv([&](rpc::Buffer *buffer, uint32_t) {
      rpc::get<Idx>(t) = *reinterpret_cast<ArgTy *>(buffer->data);
    });
  }
}

// Server-side handling of pointer arguments. We receive the memory into a
// temporary buffer and pass a pointer to this new memory back to the client.
template <uint32_t NUM_LANES, typename Tuple, uint64_t Idx>
RPC_ATTRS constexpr void prepare_arg(rpc::Server::Port &port) {
  using ArgTy = rpc::tuple_element_t<Idx, Tuple>;
  using ElemTy = rpc::remove_pointer_t<ArgTy>;
  if constexpr (rpc::is_pointer_v<ArgTy> && rpc::is_complete_v<ElemTy> &&
                !rpc::is_void_v<ElemTy>) {
    void *args[NUM_LANES]{};
    uint64_t sizes[NUM_LANES]{};
    port.recv_n(args, sizes, [](uint64_t size) {
      if constexpr (rpc::is_same_v<ArgTy, const char *>)
        return malloc(size);
      else
        return malloc(
            sizeof(rpc::remove_const_t<rpc::remove_pointer_t<ArgTy>>));
    });
    port.send([&](rpc::Buffer *buffer, uint32_t id) {
      *reinterpret_cast<ArgTy *>(buffer->data) = static_cast<ArgTy>(args[id]);
    });
  }
}

// Client-side finalization of pointer arguments. If the type is not constant we
// must copy back any potential modifications the invoked function made to that
// pointer.
template <uint64_t Idx, typename Tuple>
RPC_ATTRS constexpr void finish_arg(rpc::Client::Port &port, Tuple &t) {
  using ArgTy = rpc::tuple_element_t<Idx, Tuple>;
  using ElemTy = rpc::remove_pointer_t<ArgTy>;
  using MemoryTy = rpc::remove_const_t<rpc::remove_pointer_t<ArgTy>> *;
  if constexpr (rpc::is_pointer_v<ArgTy> && !rpc::is_const_v<ArgTy> &&
                rpc::is_complete_v<ElemTy> && !rpc::is_void_v<ElemTy>) {
    uint64_t size{};
    void *buf{};
    port.recv_n(&buf, &size, [&](uint64_t) {
      return const_cast<MemoryTy>(rpc::get<Idx>(t));
    });
  }
}

// Server-side finalization of pointer arguments. We copy any potential
// modifications to the value back to the client unless it was a constant. We
// can also free the associated memory.
template <uint32_t NUM_LANES, uint64_t Idx, typename Tuple>
RPC_ATTRS constexpr void finish_arg(rpc::Server::Port &port,
                                    Tuple (&t)[NUM_LANES]) {
  using ArgTy = rpc::tuple_element_t<Idx, Tuple>;
  using ElemTy = rpc::remove_pointer_t<ArgTy>;
  if constexpr (rpc::is_pointer_v<ArgTy> && !rpc::is_const_v<ArgTy> &&
                rpc::is_complete_v<ElemTy> && !rpc::is_void_v<ElemTy>) {
    const void *buffer[NUM_LANES]{};
    size_t sizes[NUM_LANES]{};
    for (uint32_t id = 0; id < NUM_LANES; ++id) {
      if (port.get_lane_mask() & (uint64_t(1) << id)) {
        buffer[id] = rpc::get<Idx>(t[id]);
        sizes[id] = sizeof(rpc::remove_pointer_t<ArgTy>);
      }
    }
    port.send_n(buffer, sizes);
  }

  if constexpr (rpc::is_pointer_v<ArgTy> && rpc::is_complete_v<ElemTy> &&
                !rpc::is_void_v<ElemTy>) {
    for (uint32_t id = 0; id < NUM_LANES; ++id) {
      if (port.get_lane_mask() & (uint64_t(1) << id))
        free(const_cast<void *>(
            static_cast<const void *>(rpc::get<Idx>(t[id]))));
    }
  }
}

// Iterate over the tuple list of arguments to see if we need to forward any.
// The current forwarding is somewhat inefficient as each pointer is an
// individual RPC call.
template <typename Tuple, uint64_t... Is>
RPC_ATTRS constexpr void prepare_args(rpc::Client::Port &port, Tuple &t,
                                      rpc::index_sequence<Is...>) {
  (prepare_arg<Is>(port, t), ...);
}
template <uint32_t NUM_LANES, typename Tuple, uint64_t... Is>
RPC_ATTRS constexpr void prepare_args(rpc::Server::Port &port,
                                      rpc::index_sequence<Is...>) {
  (prepare_arg<NUM_LANES, Tuple, Is>(port), ...);
}

// Performs the preparation in reverse, copying back any modified values.
template <typename Tuple, uint64_t... Is>
RPC_ATTRS constexpr void finish_args(rpc::Client::Port &port, Tuple &&t,
                                     rpc::index_sequence<Is...>) {
  (finish_arg<Is>(port, t), ...);
}
template <uint32_t NUM_LANES, typename Tuple, uint64_t... Is>
RPC_ATTRS constexpr void finish_args(rpc::Server::Port &port,
                                     Tuple (&t)[NUM_LANES],
                                     rpc::index_sequence<Is...>) {
  (finish_arg<NUM_LANES, Is>(port, t), ...);
}
} // namespace

// Dispatch a function call to the server through the RPC mechanism. Copies the
// argument list through the RPC interface.
template <uint32_t OPCODE, typename FnTy, typename... CallArgs>
RPC_ATTRS constexpr typename function_traits<FnTy>::return_type
dispatch(rpc::Client &client, FnTy, CallArgs... args) {
  using Traits = function_traits<FnTy>;
  using RetTy = typename Traits::return_type;
  using TupleTy = typename Traits::arg_types;
  using Bytes = tuple_bytes<CallArgs...>;

  static_assert(sizeof...(CallArgs) == Traits::ARITY,
                "Argument count mismatch");
  static_assert(((rpc::is_trivially_constructible_v<CallArgs> &&
                  rpc::is_trivially_copyable_v<CallArgs>) &&
                 ...),
                "Must be a trivial type");

  auto port = client.open<OPCODE>();

  // Copy over any pointer arguments by walking the argument list.
  TupleTy arg_tuple{rpc::forward<CallArgs>(args)...};
  rpc::prepare_args(port, arg_tuple, rpc::make_index_sequence<Traits::ARITY>{});

  // Compress the argument list to a binary stream and send it to the server.
  auto bytes = Bytes::pack(arg_tuple);
  port.send_n(&bytes);

  // Copy back any potentially modified pointer arguments and the return value.
  rpc::finish_args(port, TupleTy{rpc::forward<CallArgs>(args)...},
                   rpc::make_index_sequence<Traits::ARITY>{});

  // Copy back the final function return value.
  using BufferTy = rpc::conditional_t<rpc::is_void_v<RetTy>, uint8_t, RetTy>;
  BufferTy ret{};
  port.recv_n(&ret);

  if constexpr (!rpc::is_void_v<RetTy>)
    return ret;
}

// Invoke a function on the server on behalf of the client. Receives the
// arguments through the interface and forwards them to the function.
template <uint32_t NUM_LANES, typename FnTy>
RPC_ATTRS constexpr void invoke(rpc::Server::Port &port, FnTy fn) {
  using Traits = function_traits<FnTy>;
  using RetTy = typename Traits::return_type;
  using TupleTy = typename Traits::arg_types;
  using Bytes = tuple_bytes<TupleTy>;

  // Receive pointer data from the host and pack it in server-side memory.
  rpc::prepare_args<NUM_LANES, TupleTy>(
      port, rpc::make_index_sequence<Traits::ARITY>{});

  // Get the argument list from the client.
  typename Bytes::array_type arg_buf[NUM_LANES]{};
  port.recv_n(arg_buf);

  // Convert the received arguments into an invocable argument list.
  TupleTy args[NUM_LANES];
  for (uint32_t id = 0; id < NUM_LANES; ++id) {
    if (port.get_lane_mask() & (uint64_t(1) << id))
      args[id] = Bytes::unpack(arg_buf[id]);
  }

  // Execute the function with the provided arguments and send back any copies
  // made for pointer data.
  using BufferTy = rpc::conditional_t<rpc::is_void_v<RetTy>, uint8_t, RetTy>;
  BufferTy rets[NUM_LANES]{};
  for (uint32_t id = 0; id < NUM_LANES; ++id) {
    if (port.get_lane_mask() & (uint64_t(1) << id)) {
      if constexpr (rpc::is_void_v<RetTy>)
        rpc::apply(fn, args[id]);
      else
        rets[id] = rpc::apply(fn, args[id]);
    }
  }

  // Send any potentially modified pointer arguments back to the client.
  rpc::finish_args<NUM_LANES>(port, args,
                              rpc::make_index_sequence<Traits::ARITY>{});

  // Copy back the return value of the function if one exists. If the function
  // is void we send an empty packet to force synchronous behavior.
  port.send_n(rets);
}
} // namespace rpc

#endif // LLVM_LIBC_SHARED_RPC_DISPATCH_H
