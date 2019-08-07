// RUN: %clang_cc1 -fsyntax-only -Wdangling -Wdangling-field -Wreturn-stack-address -verify %s
struct [[gsl::Owner(int)]] MyIntOwner {
  MyIntOwner();
  int &operator*();
  int *c_str() const;
};

struct [[gsl::Pointer(int)]] MyIntPointer {
  MyIntPointer(int *p = nullptr);
  // Conversion operator and constructor conversion will result in two
  // different ASTs. The former is tested with another owner and 
  // pointer type.
  MyIntPointer(const MyIntOwner &);
  int &operator*();
  MyIntOwner toOwner();
};

struct [[gsl::Pointer(long)]] MyLongPointerFromConversion {
  MyLongPointerFromConversion(long *p = nullptr);
  long &operator*();
};

struct [[gsl::Owner(long)]] MyLongOwnerWithConversion {
  MyLongOwnerWithConversion();
  operator MyLongPointerFromConversion();
  long &operator*();
  MyIntPointer releaseAsMyPointer();
  long *releaseAsRawPointer();
};

void danglingHeapObject() {
  new MyLongPointerFromConversion(MyLongOwnerWithConversion{}); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  new MyIntPointer(MyIntOwner{}); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
}

void intentionalFalseNegative() {
  int i;
  MyIntPointer p{&i};
  // In this case we do not have enough information in a statement local
  // analysis to detect the problem.
  new MyIntPointer(p);
  new MyIntPointer(MyIntPointer{p});
}

MyIntPointer ownershipTransferToMyPointer() {
  MyLongOwnerWithConversion t;
  return t.releaseAsMyPointer(); // ok
}

long *ownershipTransferToRawPointer() {
  MyLongOwnerWithConversion t;
  return t.releaseAsRawPointer(); // ok
}

int *danglingRawPtrFromLocal() {
  MyIntOwner t;
  return t.c_str(); // TODO
}

int *danglingRawPtrFromTemp() {
  MyIntPointer p;
  return p.toOwner().c_str(); // TODO
}

struct Y {
  int a[4];
};

void dangligGslPtrFromTemporary() {
  MyIntPointer p = Y{}.a; // expected-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  (void)p;
}

struct DanglingGslPtrField {
  MyIntPointer p; // expected-note 2{{pointer member declared here}}
  MyLongPointerFromConversion p2; // expected-note {{pointer member declared here}}
  DanglingGslPtrField(int i) : p(&i) {} // expected-warning {{initializing pointer member 'p' with the stack address of parameter 'i'}}
  DanglingGslPtrField() : p2(MyLongOwnerWithConversion{}) {} // expected-warning {{initializing pointer member 'p2' to point to a temporary object whose lifetime is shorter than the lifetime of the constructed object}}
  DanglingGslPtrField(double) : p(MyIntOwner{}) {} // expected-warning {{initializing pointer member 'p' to point to a temporary object whose lifetime is shorter than the lifetime of the constructed object}}
};

MyIntPointer danglingGslPtrFromLocal() {
  int j;
  return &j; // expected-warning {{address of stack memory associated with local variable 'j' returned}}
}

MyIntPointer returningLocalPointer() {
  MyIntPointer localPointer;
  return localPointer; // ok
}

MyIntPointer daglingGslPtrFromLocalOwner() {
  MyIntOwner localOwner;
  return localOwner; // expected-warning {{address of stack memory associated with local variable 'localOwner' returned}}
}

MyLongPointerFromConversion daglingGslPtrFromLocalOwnerConv() {
  MyLongOwnerWithConversion localOwner;
  return localOwner; // expected-warning {{address of stack memory associated with local variable 'localOwner' returned}}
}

MyIntPointer danglingGslPtrFromTemporary() {
  return MyIntOwner{}; // expected-warning {{returning address of local temporary object}}
}

MyLongPointerFromConversion danglingGslPtrFromTemporaryConv() {
  return MyLongOwnerWithConversion{}; // expected-warning {{returning address of local temporary object}}
}

int *noFalsePositive(MyIntOwner &o) {
  MyIntPointer p = o;
  return &*p; // ok
}

MyIntPointer global;
MyLongPointerFromConversion global2;

void initLocalGslPtrWithTempOwner() {
  MyIntPointer p = MyIntOwner{}; // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  p = MyIntOwner{}; // TODO ?
  global = MyIntOwner{}; // TODO ?
  MyLongPointerFromConversion p2 = MyLongOwnerWithConversion{}; // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  p2 = MyLongOwnerWithConversion{}; // TODO ?
  global2 = MyLongOwnerWithConversion{}; // TODO ?
}

struct IntVector {
  int *begin();
  int *end();
};

void modelIterators() {
  int *it = IntVector{}.begin(); // TODO ?
  (void)it;
}
