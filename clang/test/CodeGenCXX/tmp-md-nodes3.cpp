// REQUIRES: asserts, riscv-registered-target
// Should trigger GenerateVarArgsThunk.
// RUN: %clang_cc1 -O0 -triple riscv64-linux-gnu -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN: FileCheck %s

// RUN: %clang_cc1 -O0 -triple riscv64-linux-gnu -debug-info-kind=limited -emit-obj %s -o - | \
// RUN: llvm-dwarfdump --verify -

// This test checks that clang doesn't crash when creating a varargs thunk
// by cloning a function which DILocalVariable types are unresolved at cloning
// time.
// In such case, as a workaround, instead of cloning unresolved types for
// the thunk, clang produces thunk DILocalVariables that refer to local
// types from the original DISubprogram.

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
