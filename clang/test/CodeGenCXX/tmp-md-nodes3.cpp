// REQUIRES: asserts
// Should trigger GenerateVarArgsThunk.
// RUN: %clang_cc1 -O0 -triple riscv64-linux-gnu -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN: FileCheck %s

// RUN: %clang_cc1 -O0 -triple riscv64-linux-gnu -debug-info-kind=limited -emit-obj %s -o - | \
// RUN: llvm-dwarfdump --verify -

// This test checks that the varargs thunk is correctly created if types of
// DILocalVariables of the base function are unresolved at the cloning time.

typedef signed char __int8_t;
typedef int BOOL;
class CMsgAgent;

class CFs {
public:
  typedef enum {} CACHE_HINT;
  virtual BOOL ReqCacheHint( CMsgAgent* p_ma, CACHE_HINT hint, ... ) ;
};

typedef struct {} _Lldiv_t;

class CBdVfs {
public:
  virtual ~CBdVfs( ) {}
};

class CBdVfsImpl : public CBdVfs, public CFs {
  BOOL ReqCacheHint( CMsgAgent* p_ma, CACHE_HINT hint, ... );
};

BOOL CBdVfsImpl::ReqCacheHint( CMsgAgent* p_ma, CACHE_HINT hint, ... ) {
  // Complete enum with no defintion. Clang allows to declare variables of
  // such type.
  enum class E : int;
  E enum_var;
  // Structure has forward declaration. DIDerivedType which is a pointer
  // to it is unresolved during debug info generation for the function.
  struct LocalStruct;
  LocalStruct *struct_var;

  struct GlobalStruct {};

  return true;
}

// Check that thunk is emitted.
// CHECK: define {{.*}} @_ZThn{{[48]}}_N10CBdVfsImpl12ReqCacheHintEP9CMsgAgentN3CFs10CACHE_HINTEz(
