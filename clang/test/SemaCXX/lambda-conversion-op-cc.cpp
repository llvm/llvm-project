// RUN: %clang_cc1 -fsyntax-only -triple x86_64-linux-pc %s -verify -DBAD_CONVERSION
// RUN: %clang_cc1 -fsyntax-only -triple i386-windows-pc %s -verify -DBAD_CONVERSION -DWIN32
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-linux-pc %s -ast-dump | FileCheck %s --check-prefixes=CHECK,LIN64,NODEF
// RUN: %clang_cc1 -fsyntax-only -triple i386-windows-pc %s -ast-dump -DWIN32 | FileCheck %s --check-prefixes=CHECK,WIN32,NODEF

// RUN: %clang_cc1 -fsyntax-only -triple x86_64-linux-pc -fdefault-calling-conv=vectorcall %s -verify -DBAD_VEC_CONVERS
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-linux-pc -fdefault-calling-conv=vectorcall %s -ast-dump | FileCheck %s --check-prefixes=CHECK,VECTDEF

void useage() {
  auto normal = [](int, float, double) {};                                // #1
  auto vectorcall = [](int, float, double) __attribute__((vectorcall)){}; // #2
#ifdef WIN32
  auto thiscall = [](int, float, double) __attribute__((thiscall)){}; // #3
#endif                                                                // WIN32
  auto cdecl = [](int, float, double) __attribute__((cdecl)){};

  auto genericlambda = [](auto a) {};                                      // #4
  auto genericvectorcalllambda = [](auto a) __attribute__((vectorcall)){}; // #5

  // None of these should be ambiguous.
  (void)+normal;
  (void)+vectorcall;
#ifdef WIN32
  (void)+thiscall;
#endif // WIN32
  (void)+cdecl;

#ifdef BAD_CONVERSION
  // expected-error-re@+1 {{invalid argument type {{.*}} to unary expression}}
  (void)+genericlambda;
  // expected-error-re@+1 {{invalid argument type {{.*}} to unary expression}}
  (void)+genericvectorcalllambda;
#endif // BAD_CONVERSION

  // CHECK: VarDecl {{.*}} normal '
  // CHECK: LambdaExpr
  // WIN32: CXXMethodDecl {{.*}} operator() 'void (int, float, double) __attribute__((thiscall)) const'
  // LIN64: CXXMethodDecl {{.*}} operator() 'void (int, float, double) const'
  // VECTDEF: CXXMethodDecl {{.*}} operator() 'void (int, float, double) const'
  // NODEF: CXXConversionDecl {{.*}} operator void (*)(int, float, double) 'void
  // NODEF: CXXMethodDecl {{.*}} __invoke 'void (int, float, double)' static inline
  // VECTDEF: CXXConversionDecl {{.*}} operator void (*)(int, float, double) __attribute__((vectorcall)) 'void
  // VECTDEF: CXXMethodDecl {{.*}} __invoke 'void (int, float, double) __attribute__((vectorcall))' static inline

  // CHECK: VarDecl {{.*}} vectorcall '
  // CHECK: LambdaExpr
  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int, float, double) __attribute__((vectorcall)) const'
  // CHECK: CXXConversionDecl {{.*}} operator void (*)(int, float, double) __attribute__((vectorcall)) 'void
  // CHECK: CXXMethodDecl {{.*}} __invoke 'void (int, float, double) __attribute__((vectorcall))' static inline

  // WIN32: VarDecl {{.*}} thiscall '
  // WIN32: LambdaExpr
  // WIN32: CXXMethodDecl {{.*}} operator() 'void (int, float, double) __attribute__((thiscall)) const'
  // WIN32: CXXConversionDecl {{.*}} operator void (*)(int, float, double) 'void
  // WIN32: CXXMethodDecl {{.*}} __invoke 'void (int, float, double)' static inline

  // CHECK: VarDecl {{.*}} cdecl '
  // CHECK: LambdaExpr
  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int, float, double) const'
  // NODEF: CXXConversionDecl {{.*}} operator void (*)(int, float, double) 'void
  // NODEF: CXXMethodDecl {{.*}} __invoke 'void (int, float, double)' static inline
  // VECTDEF: CXXConversionDecl {{.*}} operator void (*)(int, float, double) __attribute__((vectorcall)) 'void
  // VECTDEF: CXXMethodDecl {{.*}} __invoke 'void (int, float, double) __attribute__((vectorcall))' static inline

  // CHECK: VarDecl {{.*}} genericlambda '
  // CHECK: LambdaExpr
  //
  // CHECK: FunctionTemplateDecl {{.*}} operator()
  // LIN64: CXXMethodDecl {{.*}} operator() 'auto (auto) const' inline
  // LIN64: CXXMethodDecl {{.*}} operator() 'void (char) const' implicit_instantiation inline
  // LIN64: CXXMethodDecl {{.*}} operator() 'void (int) const' implicit_instantiation inline
  // WIN32: CXXMethodDecl {{.*}} operator() 'auto (auto) __attribute__((thiscall)) const' inline
  // WIN32: CXXMethodDecl {{.*}} operator() 'void (char) __attribute__((thiscall)) const' implicit_instantiation inline
  // WIN32: CXXMethodDecl {{.*}} operator() 'void (int) __attribute__((thiscall)) const' implicit_instantiation inline
  //
  // NODEF: FunctionTemplateDecl {{.*}} operator auto (*)(type-parameter-0-0)
  // VECDEF: FunctionTemplateDecl {{.*}} operator auto (*)(type-parameter-0-0) __attribute__((vectorcall))
  // LIN64: CXXConversionDecl {{.*}} operator auto (*)(type-parameter-0-0) 'auto (*() const noexcept)(auto)'
  // LIN64: CXXConversionDecl {{.*}} operator auto (*)(char) 'void (*() const noexcept)(char)'
  // LIN64: CXXConversionDecl {{.*}} operator auto (*)(int) 'void (*() const noexcept)(int)'
  // WIN32: CXXConversionDecl {{.*}} operator auto (*)(type-parameter-0-0) 'auto (*() __attribute__((thiscall)) const noexcept)(auto)'
  // WIN32: CXXConversionDecl {{.*}} operator auto (*)(char) 'void (*() __attribute__((thiscall)) const noexcept)(char)'
  // WIN32: CXXConversionDecl {{.*}} operator auto (*)(int) 'void (*() __attribute__((thiscall)) const noexcept)(int)'
  // VECDEF: CXXConversionDecl {{.*}} operator auto (*)(type-parameter-0-0) __attribute__((vectorcall)) 'auto (*() const noexcept)(auto)' __attribute__((vectorcall))
  // VECDEF: CXXConversionDecl {{.*}} operator auto (*)(char) __attribute__((vectorcall)) 'void (*() const noexcept)(char)' __attribute__((vectorcall))
  // VECDEF: CXXConversionDecl {{.*}} operator auto (*)(int) __attribute__((vectorcall)) 'void (*() const noexcept)(int)' __attribute__((vectorcall))
  //
  // CHECK: FunctionTemplateDecl {{.*}} __invoke
  // NODEF: CXXMethodDecl {{.*}} __invoke 'auto (auto)'
  // NODEF: CXXMethodDecl {{.*}} __invoke 'void (char)'
  // NODEF: CXXMethodDecl {{.*}} __invoke 'void (int)'
  // VECDEF: CXXMethodDecl {{.*}} __invoke 'auto (auto) __attribute__((vectorcall))'
  // VECDEF: CXXMethodDecl {{.*}} __invoke 'void (char) __attribute__((vectorcall))'
  // VECDEF: CXXMethodDecl {{.*}} __invoke 'void (int) __attribute__((vectorcall))'
  //
  // ONLY WIN32 has the duplicate here.
  // WIN32: FunctionTemplateDecl {{.*}} operator auto (*)(type-parameter-0-0) __attribute__((thiscall))
  // WIN32: CXXConversionDecl {{.*}} operator auto (*)(type-parameter-0-0) __attribute__((thiscall)) 'auto (*() __attribute__((thiscall)) const noexcept)(auto) __attribute__((thiscall))'
  // WIN32: CXXConversionDecl {{.*}} operator auto (*)(char) __attribute__((thiscall)) 'void (*() __attribute__((thiscall)) const noexcept)(char) __attribute__((thiscall))'
  // WIN32: CXXConversionDecl {{.*}} operator auto (*)(int) __attribute__((thiscall)) 'void (*() __attribute__((thiscall)) const noexcept)(int) __attribute__((thiscall))'
  //
  // WIN32: FunctionTemplateDecl {{.*}} __invoke
  // WIN32: CXXMethodDecl {{.*}} __invoke 'auto (auto) __attribute__((thiscall))'
  // WIN32: CXXMethodDecl {{.*}} __invoke 'void (char) __attribute__((thiscall))'
  // WIN32: CXXMethodDecl {{.*}} __invoke 'void (int) __attribute__((thiscall))'

  // CHECK: VarDecl {{.*}} genericvectorcalllambda '
  // CHECK: LambdaExpr
  // CHECK: FunctionTemplateDecl {{.*}} operator()
  // CHECK: CXXMethodDecl {{.*}} operator() 'auto (auto) __attribute__((vectorcall)) const' inline
  // CHECK: CXXMethodDecl {{.*}} operator() 'void (char) __attribute__((vectorcall)) const' implicit_instantiation inline
  // CHECK: CXXMethodDecl {{.*}} operator() 'void (int) __attribute__((vectorcall)) const' implicit_instantiation inline
  // CHECK: FunctionTemplateDecl {{.*}} operator auto (*)(type-parameter-0-0) __attribute__((vectorcall))
  // LIN64: CXXConversionDecl {{.*}} operator auto (*)(type-parameter-0-0) __attribute__((vectorcall)) 'auto (*() const noexcept)(auto) __attribute__((vectorcall))'
  // LIN64: CXXConversionDecl {{.*}} operator auto (*)(char) __attribute__((vectorcall)) 'void (*() const noexcept)(char) __attribute__((vectorcall))'
  // LIN64: CXXConversionDecl {{.*}} operator auto (*)(int) __attribute__((vectorcall)) 'void (*() const noexcept)(int) __attribute__((vectorcall))'
  // WIN32: CXXConversionDecl {{.*}} operator auto (*)(type-parameter-0-0) __attribute__((vectorcall)) 'auto (*() __attribute__((thiscall)) const noexcept)(auto) __attribute__((vectorcall))'
  // WIN32: CXXConversionDecl {{.*}} operator auto (*)(char) __attribute__((vectorcall)) 'void (*() __attribute__((thiscall)) const noexcept)(char) __attribute__((vectorcall))'
  // WIN32: CXXConversionDecl {{.*}} operator auto (*)(int) __attribute__((vectorcall)) 'void (*() __attribute__((thiscall)) const noexcept)(int) __attribute__((vectorcall))'
  // CHECK: FunctionTemplateDecl {{.*}} __invoke
  // CHECK: CXXMethodDecl {{.*}} __invoke 'auto (auto) __attribute__((vectorcall))'
  // CHECK: CXXMethodDecl {{.*}} __invoke 'void (char) __attribute__((vectorcall))'
  // CHECK: CXXMethodDecl {{.*}} __invoke 'void (int) __attribute__((vectorcall))'

  // NODEF: UnaryOperator {{.*}} 'void (*)(int, float, double)' prefix '+'
  // NODEF-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int, float, double)'
  // NODEF-NEXT: CXXMemberCallExpr {{.*}}'void (*)(int, float, double)'
  // VECTDEF: UnaryOperator {{.*}} 'void (*)(int, float, double) __attribute__((vectorcall))' prefix '+'
  // VECTDEF-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int, float, double) __attribute__((vectorcall))'
  // VECTDEF-NEXT: CXXMemberCallExpr {{.*}}'void (*)(int, float, double) __attribute__((vectorcall))'

  // CHECK: UnaryOperator {{.*}} 'void (*)(int, float, double) __attribute__((vectorcall))' prefix '+'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int, float, double) __attribute__((vectorcall))'
  // CHECK-NEXT: CXXMemberCallExpr {{.*}}'void (*)(int, float, double) __attribute__((vectorcall))'

  // WIN32: UnaryOperator {{.*}} 'void (*)(int, float, double)' prefix '+'
  // WIN32-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int, float, double)'
  // WIN32-NEXT: CXXMemberCallExpr {{.*}}'void (*)(int, float, double)'

  // NODEF: UnaryOperator {{.*}} 'void (*)(int, float, double)' prefix '+'
  // NODEF-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int, float, double)'
  // NODEF-NEXT: CXXMemberCallExpr {{.*}}'void (*)(int, float, double)'
  // VECTDEF: UnaryOperator {{.*}} 'void (*)(int, float, double) __attribute__((vectorcall))' prefix '+'
  // VECTDEF-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int, float, double) __attribute__((vectorcall))'
  // VECTDEF-NEXT: CXXMemberCallExpr {{.*}}'void (*)(int, float, double) __attribute__((vectorcall))'

#ifdef BAD_CONVERSION
  // expected-error-re@+2 {{no viable conversion from {{.*}} to 'void (*)(int, float, double) __attribute__((vectorcall))}}
  // expected-note@#1 {{candidate function}}
  void (*__attribute__((vectorcall)) normal_ptr2)(int, float, double) = normal;
  // expected-error-re@+2 {{no viable conversion from {{.*}} to 'void (*)(int, float, double)}}
  // expected-note@#2 {{candidate function}}
  void (*vectorcall_ptr2)(int, float, double) = vectorcall;
#ifdef WIN32
  void (*__attribute__((thiscall)) thiscall_ptr2)(int, float, double) = thiscall;
#endif // WIN32
  // expected-error-re@+2 {{no viable conversion from {{.*}} to 'void (*)(char) __attribute__((vectorcall))'}}
  // expected-note@#4 {{candidate function}}
  void(__vectorcall * generic_ptr)(char) = genericlambda;
  // expected-error-re@+2 {{no viable conversion from {{.*}} to 'void (*)(char)}}
  // expected-note@#5 {{candidate function}}
  void (*generic_ptr2)(char) = genericvectorcalllambda;
#endif // BAD_CONVERSION

#ifdef BAD_VEC_CONVERS
  void (*__attribute__((vectorcall)) normal_ptr2)(int, float, double) = normal;
  void (*normal_ptr3)(int, float, double) = normal;
  // expected-error-re@+2 {{no viable conversion from {{.*}} to 'void (*)(int, float, double) __attribute__((regcall))}}
  // expected-note@#1 {{candidate function}}
  void (*__attribute__((regcall)) normalptr4)(int, float, double) = normal;
  void (*__attribute__((vectorcall)) vectorcall_ptr2)(int, float, double) = vectorcall;
  void (*vectorcall_ptr3)(int, float, double) = vectorcall;
#endif // BAD_VEC_CONVERS

  // Required to force emission of the invoker.
  void (*normal_ptr)(int, float, double) = normal;
  void (*__attribute__((vectorcall)) vectorcall_ptr)(int, float, double) = vectorcall;
#ifdef WIN32
  void (*thiscall_ptr)(int, float, double) = thiscall;
#endif // WIN32
  void (*cdecl_ptr)(int, float, double) = cdecl;
  void (*generic_ptr3)(char) = genericlambda;
  void (*generic_ptr4)(int) = genericlambda;
#ifdef WIN32
  void (*__attribute__((thiscall)) generic_ptr3b)(char) = genericlambda;
  void (*__attribute__((thiscall)) generic_ptr4b)(int) = genericlambda;
#endif
  void (*__attribute__((vectorcall)) generic_ptr5)(char) = genericvectorcalllambda;
  void (*__attribute__((vectorcall)) generic_ptr6)(int) = genericvectorcalllambda;
}
