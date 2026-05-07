struct A {
  virtual void f();
  int a;
};

struct B {
  virtual void g();
  int b;
};

struct C : A, B {
  void f() override;
  void g() override;
  int c;
};

void A::f() {}
void B::g() {}
void C::f() {}
void C::g() {}

// 1. 通过 A* 做虚调用。
//    这里应该会查 A-subobject 的 vptr，然后从 vtable slot 里取 f。
void call_through_A(A *pa) {
  pa->f();
}

// 2. 通过 B* 做虚调用。
//    这里应该会查 B-subobject 的 vptr，然后从 secondary vtable slot 里取 g。
//    如果实际对象是 C，slot 里可能是 thunk。
void call_through_B(B *pb) {
  pb->g();
}

// 3. 已知静态类型是 C*，但调用 virtual 函数。
//    C++ 语义上仍然是 virtual call，除非编译器能 devirtualize。
//    在 -O0 / CIR 阶段通常更容易看到 vptr load。
void call_through_C(C *pc) {
  pc->f();
  pc->g();
}

// 4. 非虚的限定调用。
//    这里不会查 vptr，应该直接 call C::f / C::g。
void direct_qualified_call(C *pc) {
  pc->C::f();
  pc->C::g();
}

// 5. 从 C* 转成 A* / B*。
//    A 是 primary base，一般 offset 是 0。
//    B 是 secondary base，一般需要 this + 16 之类的调整。
void base_casts(C *pc) {
  A *pa = pc;
  B *pb = pc;

  pa->f();
  pb->g();
}

// 6. 栈上构造 C。
//    这里适合观察 constructor / vptr 初始化。
//    如果 CIR 暂时没有完整 ctor lowering，也至少能暴露相关路径。
void construct_and_call() {
  C obj;
  obj.f();
  obj.g();

  A *pa = &obj;
  B *pb = &obj;
  pa->f();
  pb->g();
}