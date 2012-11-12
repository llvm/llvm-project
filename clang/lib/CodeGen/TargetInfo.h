//===---- TargetInfo.h - Encapsulate target details -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_TARGETINFO_H
#define CLANG_CODEGEN_TARGETINFO_H

#include "clang/Basic/LLVM.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
  class GlobalValue;
  class Type;
  class Value;
}

namespace clang {
  class ABIInfo;
  class Decl;

  namespace CodeGen {
    class CallArgList;
    class CodeGenModule;
    class CodeGenFunction;
    class CGFunctionInfo;
  }

  /// TargetCodeGenInfo - This class organizes various target-specific
  /// codegeneration issues, like target-specific attributes, builtins and so
  /// on.
  class TargetCodeGenInfo {
    ABIInfo *Info;
  public:
    // WARNING: Acquires the ownership of ABIInfo.
    TargetCodeGenInfo(ABIInfo *info = 0):Info(info) { }
    virtual ~TargetCodeGenInfo();

    /// getABIInfo() - Returns ABI info helper for the target.
    const ABIInfo& getABIInfo() const { return *Info; }

    /// SetTargetAttributes - Provides a convenient hook to handle extra
    /// target-specific attributes for the given global.
    virtual void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                                     CodeGen::CodeGenModule &M) const { }

    /// Determines the size of struct _Unwind_Exception on this platform,
    /// in 8-bit units.  The Itanium ABI defines this as:
    ///   struct _Unwind_Exception {
    ///     uint64 exception_class;
    ///     _Unwind_Exception_Cleanup_Fn exception_cleanup;
    ///     uint64 private_1;
    ///     uint64 private_2;
    ///   };
    virtual unsigned getSizeOfUnwindException() const;

    /// Controls whether __builtin_extend_pointer should sign-extend
    /// pointers to uint64_t or zero-extend them (the default).  Has
    /// no effect for targets:
    ///   - that have 64-bit pointers, or
    ///   - that cannot address through registers larger than pointers, or
    ///   - that implicitly ignore/truncate the top bits when addressing
    ///     through such registers.
    virtual bool extendPointerWithSExt() const { return false; }

    /// Determines the DWARF register number for the stack pointer, for
    /// exception-handling purposes.  Implements __builtin_dwarf_sp_column.
    ///
    /// Returns -1 if the operation is unsupported by this target.
    virtual int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
      return -1;
    }

    /// Initializes the given DWARF EH register-size table, a char*.
    /// Implements __builtin_init_dwarf_reg_size_table.
    ///
    /// Returns true if the operation is unsupported by this target.
    virtual bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                                         llvm::Value *Address) const {
      return true;
    }

    /// Performs the code-generation required to convert a return
    /// address as stored by the system into the actual address of the
    /// next instruction that will be executed.
    ///
    /// Used by __builtin_extract_return_addr().
    virtual llvm::Value *decodeReturnAddress(CodeGen::CodeGenFunction &CGF,
                                             llvm::Value *Address) const {
      return Address;
    }

    /// Performs the code-generation required to convert the address
    /// of an instruction into a return address suitable for storage
    /// by the system in a return slot.
    ///
    /// Used by __builtin_frob_return_addr().
    virtual llvm::Value *encodeReturnAddress(CodeGen::CodeGenFunction &CGF,
                                             llvm::Value *Address) const {
      return Address;
    }

    virtual llvm::Type* adjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                            StringRef Constraint, 
                                            llvm::Type* Ty) const {
      return Ty;
    }

    /// Retrieve the address of a function to call immediately before
    /// calling objc_retainAutoreleasedReturnValue.  The
    /// implementation of objc_autoreleaseReturnValue sniffs the
    /// instruction stream following its return address to decide
    /// whether it's a call to objc_retainAutoreleasedReturnValue.
    /// This can be prohibitively expensive, depending on the
    /// relocation model, and so on some targets it instead sniffs for
    /// a particular instruction sequence.  This functions returns
    /// that instruction sequence in inline assembly, which will be
    /// empty if none is required.
    virtual StringRef getARCRetainAutoreleasedReturnValueMarker() const {
      return "";
    }

    /// Determine whether a call to an unprototyped functions under
    /// the given calling convention should use the variadic
    /// convention or the non-variadic convention.
    ///
    /// There's a good reason to make a platform's variadic calling
    /// convention be different from its non-variadic calling
    /// convention: the non-variadic arguments can be passed in
    /// registers (better for performance), and the variadic arguments
    /// can be passed on the stack (also better for performance).  If
    /// this is done, however, unprototyped functions *must* use the
    /// non-variadic convention, because C99 states that a call
    /// through an unprototyped function type must succeed if the
    /// function was defined with a non-variadic prototype with
    /// compatible parameters.  Therefore, splitting the conventions
    /// makes it impossible to call a variadic function through an
    /// unprototyped type.  Since function prototypes came out in the
    /// late 1970s, this is probably an acceptable trade-off.
    /// Nonetheless, not all platforms are willing to make it, and in
    /// particularly x86-64 bends over backwards to make the
    /// conventions compatible.
    ///
    /// The default is false.  This is correct whenever:
    ///   - the conventions are exactly the same, because it does not
    ///     matter and the resulting IR will be somewhat prettier in
    ///     certain cases; or
    ///   - the conventions are substantively different in how they pass
    ///     arguments, because in this case using the variadic convention
    ///     will lead to C99 violations.
    /// It is not necessarily correct when arguments are passed in the
    /// same way and some out-of-band information is passed for the
    /// benefit of variadic callees, as is the case for x86-64.
    /// In this case the ABI should be consulted.
    virtual bool isNoProtoCallVariadic(const CodeGen::CallArgList &args,
                                       const FunctionNoProtoType *fnType) const;
  };
}

#endif // CLANG_CODEGEN_TARGETINFO_H
