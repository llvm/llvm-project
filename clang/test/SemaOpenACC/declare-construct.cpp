// RUN: %clang_cc1 %s -fopenacc -verify

int *Global;
int GlobalArray[5];
// expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare
namespace NS {
int *NSVar;
int NSArray[5];
// expected-error@+2{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(Global, GlobalArray)
// Ok, correct scope.
#pragma acc declare create(NSVar, NSArray)

// expected-error@+4{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-3{{previous reference is here}}
// expected-error@+2{{variable referenced in 'copyin' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-5{{previous reference is here}}
#pragma acc declare create(NSVar) copyin(NSVar)

// expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare

int NSVar1, NSVar2, NSVar3, NSVar4, NSVar5, *NSVar6, NSVar7, NSVar8;

// Only create, copyin, deviceptr, device-resident, link at NS scope.
// expected-error@+3{{OpenACC 'copy' clause on a 'declare' directive is not allowed at global or namespace scope}}
// expected-error@+2{{OpenACC 'copyout' clause on a 'declare' directive is not allowed at global or namespace scope}}
// expected-error@+1{{OpenACC 'present' clause on a 'declare' directive is not allowed at global or namespace scope}}
#pragma acc declare copy(NSVar1) copyin(NSVar2), copyout(NSVar3), create(NSVar4), present(NSVar5), deviceptr(NSVar6), device_resident(NSVar7), link(NSVar8)

extern "C" {
  int ExternVar, ExternVar1, ExternVar2, ExternVar3, ExternVar4, *ExternVar5, ExternVar6, ExternVar7;
  // Only create, copyin, deviceptr, device-resident, link at NS scope.
  // expected-error@+3{{OpenACC 'copy' clause on a 'declare' directive is not allowed at global or namespace scope}}
  // expected-error@+2{{OpenACC 'copyout' clause on a 'declare' directive is not allowed at global or namespace scope}}
  // expected-error@+1{{OpenACC 'present' clause on a 'declare' directive is not allowed at global or namespace scope}}
#pragma acc declare copy(ExternVar) copyin(ExternVar1), copyout(ExternVar2), create(ExternVar3), present(ExternVar4), deviceptr(ExternVar5), device_resident(ExternVar6), link(ExternVar7)
  }
}
// expected-error@+2{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(NS::NSVar, NS::NSArray)

struct Struct {
  static const int StaticMem = 5;
  static const int StaticMem2 = 5;
  int NonStaticMem;
// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(Global)
  // OK, same scope.
#pragma acc declare create(StaticMem, StaticMem2)
// expected-error@+4{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-2{{previous reference is here}}
// expected-error@+2{{variable referenced in 'copyin' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-4{{previous reference is here}}
#pragma acc declare create(StaticMem) copyin(StaticMem)
// expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare

  void Inline(int Arg) {
// expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare
// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(StaticMem)

    int Local, Local2, Local3, Local4;
  // OK, same scope.
#pragma acc declare create(Local, Arg)
// expected-error@+2{{variable referenced in 'copyin' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@+1{{previous reference is here}}
#pragma acc declare create(Local2) copyin(Local2)

    for (int I = 0; I < 5; ++I) {
      int Other;
    // FIXME: We don't catch this because we use decl-context instead of scope.
#pragma acc declare create(Local3, Local4)
      // OK, same scope.
#pragma acc declare create(I, Other)
// expected-error@+4 2{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-2 2{{previous reference is here}}
// expected-error@+2 2{{variable referenced in 'copyin' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-4 2{{previous reference is here}}
#pragma acc declare create(I, Other) copyin(I, Other)

// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(NonStaticMem)
    }
  }

  void OutOfLine(int Arg, int Arg2);
};

void Struct::OutOfLine(int Arg, int Arg2) {
// expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare
// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(StaticMem)

  int Local, Local2;
// OK, same scope.
#pragma acc declare create(Local, Arg)
// expected-error@+4{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-2{{previous reference is here}}
// expected-error@+2{{variable referenced in 'copyin' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-4{{previous reference is here}}
#pragma acc declare create(Local) copyin(Local)

  for (int I = 0; I < 5; ++I) {
    int Other;
    // FIXME: We don't catch this because we use decl-context instead of scope.
#pragma acc declare create(Local2, Arg2)
    // OK, same scope.
#pragma acc declare create(I, Other)
// expected-error@+4 2{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-2 2{{previous reference is here}}
// expected-error@+2 2{{variable referenced in 'copyin' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-4 2{{previous reference is here}}
#pragma acc declare create(I, Other) copyin(I, Other)
  }
// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(NonStaticMem)
}

template<typename T>
struct DepStruct {
  static const T DepStaticMem = 5;
  static const int StaticMem = 5;
  int NonStaticMem;
// expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare
// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(Global)
  // OK, same scope.
#pragma acc declare create(DepStaticMem)
  // OK, same scope.
#pragma acc declare create(StaticMem)
// expected-error@+4{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-2{{previous reference is here}}
// expected-error@+2{{variable referenced in 'copyin' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-4{{previous reference is here}}
#pragma acc declare create(StaticMem) copyin(StaticMem)
// expected-error@+4{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-9{{previous reference is here}}
// expected-error@+2{{variable referenced in 'copyin' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-11{{previous reference is here}}
#pragma acc declare create(DepStaticMem) copyin(DepStaticMem)

  void Inline(int Arg) {
// expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare
// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(DepStaticMem)
// expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(StaticMem)

    T Local, Local2;
  // OK, same scope.
#pragma acc declare create(Local, Arg)
// expected-error@+2 2{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-2 2{{previous reference is here}}
#pragma acc declare create(Local, Local)

    for (int I = 0; I < 5; ++I) {
      int Other;
      // FIXME: Since we approximate this as a decl-context, we can't check
      // scope here.
#pragma acc declare create(Local2)
      // OK, same scope.
#pragma acc declare create(I, Other)
      // expected-error@+2 3{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
      // expected-note@-2 3{{previous reference is here}}
#pragma acc declare create(I, Other, I)
    }
    // expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(NonStaticMem)
  }

  void OutOfLine(int Arg);

  template<typename U>
  void TemplInline(U Arg, U Arg2) {
// expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare
    // expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(DepStaticMem)
    // expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(StaticMem)

    T Local, Local2, Local3;
  // OK, same scope.
#pragma acc declare create(Local, Arg)
// expected-error@+4{{variable referenced in 'create' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-2{{previous reference is here}}
// expected-error@+2{{variable referenced in 'present' clause of OpenACC 'declare' directive was already referenced}}
// expected-note@-4{{previous reference is here}}
#pragma acc declare create(Local2, Arg) present(Local, Arg2)
    {
      // FIXME: We don't catch this, since we check decl-context not scopes.
#pragma acc declare create(Local3)

      // expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(NonStaticMem)
    }
  }
  template<typename U>
  void TemplOutline(U Arg);
};

template<typename T>
void DepStruct<T>::OutOfLine(int Arg) {
  // expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare
  // expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(StaticMem)

  T Local, Local2;
// OK, same scope.
#pragma acc declare create(Local, Arg)

  for (int I = 0; I < 5; ++I) {
    int Other;
    // FIXME: We don't catch this because we use decl-context instead of scope.
#pragma acc declare create(Local2)
    // OK, same scope.
#pragma acc declare create(I, Other)
  }
  // expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(NonStaticMem)
}
template<typename T>
template<typename U>
void DepStruct<T>::TemplOutline(U Arg) {
// expected-error@+1{{no valid clauses specified in OpenACC 'declare' directive}}
#pragma acc declare
  // expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(DepStaticMem)
  // expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(StaticMem)

  T Local, Local2;
// OK, same scope.
#pragma acc declare create(Local, Arg)

  {
    // FIXME: We could potentially fix this someday, but as we don't have
    // 'scope' information like this during template instantiation, we have to
    // permit this.
#pragma acc declare create(Local2)
  }
  // expected-error@+1{{variable appearing in 'create' clause of OpenACC 'declare' directive must be in the same scope as the directive}}
#pragma acc declare create(NonStaticMem)
}

void use() {
  DepStruct<int> DS;
  DS.Inline(1);
  DS.OutOfLine(1);
  DS.TemplInline(1, 2);
  DS.TemplOutline(1);
}

// Only variable or array name.

// expected-error@+1{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
#pragma acc declare create(GlobalArray[0])
// expected-error@+1{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
#pragma acc declare create(GlobalArray[0: 1])

struct S { int I; };
// expected-error@+1{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
#pragma acc declare create(S{}.I)

int GS1, GS2, GS3, GS4, GS5, *GS6, GS7, GS8;

// Only create, copyin, deviceptr, device-resident, link at NS scope.
// expected-error@+3{{OpenACC 'copy' clause on a 'declare' directive is not allowed at global or namespace scope}}
// expected-error@+2{{OpenACC 'copyout' clause on a 'declare' directive is not allowed at global or namespace scope}}
// expected-error@+1{{OpenACC 'present' clause on a 'declare' directive is not allowed at global or namespace scope}}
#pragma acc declare copy(GS1) copyin(GS2), copyout(GS3), create(GS4), present(GS5), deviceptr(GS6), device_resident(GS7), link(GS8)

void ExternVar() {
  extern int I, I2, I3, I4, I5, *I6, I7, I8;
// expected-error@+3{{'extern' variable may not be referenced by 'copy' clause on an OpenACC 'declare' directive}}
// expected-error@+2{{'extern' variable may not be referenced by 'copyout' clause on an OpenACC 'declare' directive}}
// expected-error@+1{{'extern' variable may not be referenced by 'present' clause on an OpenACC 'declare' directive}}
#pragma acc declare copy(I) copyin(I2), copyout(I3), create(I4), present(I5), deviceptr(I6), device_resident(I7), link(I8)
}

// Link can only have global, namespace, or extern vars.
#pragma acc declare link(Global, GlobalArray)

struct Struct2 {
  static const int StaticMem = 5;
  // expected-error@+1{{variable referenced by 'link' clause not in global or namespace scope must be marked 'extern'}}
#pragma acc declare link(StaticMem)

  void MemFunc(int I) {
    int Local;
    extern int ExternLocal;

  // expected-error@+2{{variable referenced by 'link' clause not in global or namespace scope must be marked 'extern'}}
  // expected-error@+1{{variable referenced by 'link' clause not in global or namespace scope must be marked 'extern'}}
#pragma acc declare link(I, Local, ExternLocal)
}
};

void ModList() {
  int V1, V2, V3, V4, V4B, V5, V6, V7, V7B, V8, V9, V10,
      V11, V11B, V12, V13, V14, V15, V16, V17, V18, V19;
  // expected-error@+2{{OpenACC 'readonly' modifier not valid on 'copy' clause}}
  // expected-error@+1{{OpenACC 'zero' modifier not valid on 'copy' clause}}
#pragma acc declare copy(always, alwaysin, alwaysout, zero, readonly: V1)
  // expected-error@+1{{OpenACC 'readonly' modifier not valid on 'copy' clause}}
#pragma acc declare copy(readonly: V2)
  // expected-error@+1{{OpenACC 'zero' modifier not valid on 'copy' clause}}
#pragma acc declare copy(zero: V3)
#pragma acc declare copy(capture: V4)
#pragma acc declare copy(always, alwaysin, alwaysout, capture: V4B)

  // expected-error@+2{{OpenACC 'alwaysout' modifier not valid on 'copyin' clause}}
  // expected-error@+1{{OpenACC 'zero' modifier not valid on 'copyin' clause}}
#pragma acc declare copyin(always, alwaysin, alwaysout, zero, readonly: V5)
  // expected-error@+1{{OpenACC 'alwaysout' modifier not valid on 'copyin' clause}}
#pragma acc declare copyin(alwaysout: V6)
  // expected-error@+1{{OpenACC 'zero' modifier not valid on 'copyin' clause}}
#pragma acc declare copyin(zero: V7)
  // expected-error@+1{{OpenACC 'capture' modifier not valid on 'copyin' clause}}
#pragma acc declare copyin(capture: V7B)
  // expected-error@+1{{OpenACC 'capture' modifier not valid on 'copyin' clause}}
#pragma acc declare copyin(always, alwaysin, readonly, capture: V8)

  // expected-error@+2{{OpenACC 'alwaysin' modifier not valid on 'copyout' clause}}
  // expected-error@+1{{OpenACC 'readonly' modifier not valid on 'copyout' clause}}
#pragma acc declare copyout(always, alwaysin, alwaysout, zero, readonly: V9)
  // expected-error@+1{{OpenACC 'alwaysin' modifier not valid on 'copyout' clause}}
#pragma acc declare copyout(alwaysin: V10)
  // expected-error@+1{{OpenACC 'readonly' modifier not valid on 'copyout' clause}}
#pragma acc declare copyout(readonly: V11)
  // expected-error@+1{{OpenACC 'capture' modifier not valid on 'copyout' clause}}
#pragma acc declare copyout(capture: V11B)
  // expected-error@+2{{OpenACC 'alwaysin' modifier not valid on 'copyout' clause}}
  // expected-error@+1{{OpenACC 'capture' modifier not valid on 'copyout' clause}}
#pragma acc declare copyout(always, alwaysin, alwaysout, zero, capture: V12)

  // expected-error@+5{{OpenACC 'always' modifier not valid on 'create' clause}}
  // expected-error@+4{{OpenACC 'alwaysin' modifier not valid on 'create' clause}}
  // expected-error@+3{{OpenACC 'alwaysout' modifier not valid on 'create' clause}}
  // expected-error@+2{{OpenACC 'readonly' modifier not valid on 'create' clause}}
  // expected-error@+1{{OpenACC 'capture' modifier not valid on 'create' clause}}
#pragma acc declare create(always, alwaysin, alwaysout, zero, readonly, capture: V13)
  // expected-error@+1{{OpenACC 'always' modifier not valid on 'create' clause}}
#pragma acc declare create(always: V14)
  // expected-error@+1{{OpenACC 'alwaysin' modifier not valid on 'create' clause}}
#pragma acc declare create(alwaysin: V15)
  // expected-error@+1{{OpenACC 'alwaysout' modifier not valid on 'create' clause}}
#pragma acc declare create(alwaysout: V16)
  // expected-error@+1{{OpenACC 'readonly' modifier not valid on 'create' clause}}
#pragma acc declare create(readonly: V17)

#pragma acc declare create(zero: V18)
  // expected-error@+1{{OpenACC 'capture' modifier not valid on 'create' clause}}
#pragma acc declare create(capture: V19)
}
