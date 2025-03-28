#ifndef ABIFUNCTIONINFO_H
#define ABIFUNCTIONINFO_H

#include "LLVMABI/Type.h"
#include <vector>

// Does the leaf leave the tree or the tree let go of the leaf?
using namespace ABI;

namespace ABIFunction{

/// ABIArgInfo - Helper class to encapsulate information about how a
/// specific C type should be passed to or returned from a function.
class ABIArgInfo {
public:
  enum Kind : uint8_t {
    /// Direct - Pass the argument directly using the normal converted LLVM
    /// type, or by coercing to another specified type stored in
    /// 'CoerceToType').  If an offset is specified (in UIntData), then the
    /// argument passed is offset by some number of bytes in the memory
    /// representation. A dummy argument is emitted before the real argument
    /// if the specified type stored in "PaddingType" is not zero.
    Direct,

    /// Extend - Valid only for integer argument types. Same as 'direct'
    /// but also emit a zero/sign extension attribute.
    Extend,

    /// Indirect - Pass the argument indirectly via a hidden pointer with the
    /// specified alignment (0 indicates default alignment) and address space.
    Indirect,

    /// IndirectAliased - Similar to Indirect, but the pointer may be to an
    /// object that is otherwise referenced.  The object is known to not be
    /// modified through any other references for the duration of the call, and
    /// the callee must not itself modify the object.  Because C allows
    /// parameter variables to be modified and guarantees that they have unique
    /// addresses, the callee must defensively copy the object into a local
    /// variable if it might be modified or its address might be compared.
    /// Since those are uncommon, in principle this convention allows programs
    /// to avoid copies in more situations.  However, it may introduce *extra*
    /// copies if the callee fails to prove that a copy is unnecessary and the
    /// caller naturally produces an unaliased object for the argument.
    IndirectAliased,

    /// Ignore - Ignore the argument (treat as void). Useful for void and
    /// empty structs.
    Ignore,

    /// Expand - Only valid for aggregate argument types. The structure should
    /// be expanded into consecutive arguments for its constituent fields.
    /// Currently expand is only allowed on structures whose fields
    /// are all scalar types or are themselves expandable types.
    Expand,

    /// CoerceAndExpand - Only valid for aggregate argument types. The
    /// structure should be expanded into consecutive arguments corresponding
    /// to the non-array elements of the type stored in CoerceToType.
    /// Array elements in the type are assumed to be padding and skipped.
    CoerceAndExpand,

    /// InAlloca - Pass the argument directly using the LLVM inalloca attribute.
    /// This is similar to indirect with byval, except it only applies to
    /// arguments stored in memory and forbids any implicit copies.  When
    /// applied to a return type, it means the value is returned indirectly via
    /// an implicit sret parameter stored in the argument struct.
    InAlloca,
    KindFirst = Direct,
    KindLast = InAlloca
  };

private:
  Kind TheKind;
 
public:
  ABIArgInfo(Kind K = Direct) : TheKind(K) {}
        
  Kind getKind() const { return TheKind; }

  // TODO
  void dump() const;
};

// taken from clang, needs to be extended.
enum CallingConv {
    CC_C,                  // __attribute__((cdecl))
    CC_X86StdCall       // __attribute__((stdcall))
};

struct ABIFunctionInfoArgInfo {
  ABIQualType type;
  ABIArgInfo info;
};

class ABIFunctionInfo {
  typedef ABIFunctionInfoArgInfo ArgInfo;

  private:
  CallingConv CC;
  std::vector<ArgInfo> Parameters;
  ArgInfo RetInfo;
  
  public:
    ABIFunctionInfo(CallingConv cc, std::vector<ABIQualType> parameters, ABIQualType ReturnInfo);

    static ABIFunctionInfo *
      create(CallingConv cc, std::vector<ABIQualType> parameters, ABIQualType ReturnInfo);

    ABIArgInfo &ABIFunctionInfo::getReturnInfo() { return RetInfo.info; }
    ABIQualType &ABIFunctionInfo::getReturnType() { return RetInfo.type; }

    using arg_iterator = std::vector<ArgInfo>::iterator;

    arg_iterator ABIFunctionInfo::arg_begin() { return Parameters.begin(); }
    arg_iterator ABIFunctionInfo::arg_end() { return Parameters.end(); }

    unsigned getCallingConvention() const { return CC; }

};

}
