// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
struct Data { };
struct T {
  Data *begin();
  Data *end();
};

struct NoBegin {
  Data *end();
};

struct DeletedEnd : public T {
  Data *begin();
  Data *end() = delete; //expected-note {{function has been explicitly marked deleted here}}
};

struct DeletedADLBegin { };

int* begin(DeletedADLBegin) = delete; //expected-note {{candidate function has been explicitly deleted}} \
 expected-note 5 {{candidate function not viable: no known conversion}}

struct PrivateEnd {
  Data *begin();

 private:
  Data *end(); // expected-note 2 {{declared private here}}
};

struct ADLNoEnd { };
Data * begin(ADLNoEnd); // expected-note 6 {{candidate function not viable: no known conversion}}

struct OverloadedStar {
  T operator*();
};

void f() {
  T t;
  for (auto i : t) { }
  T *pt;
  for (auto i : pt) { } // expected-error{{invalid range expression of type 'T *'; did you mean to dereference it with '*'?}}

  int arr[10];
  for (auto i : arr) { }
  int (*parr)[10];
  for (auto i : parr) { }// expected-error{{invalid range expression of type 'int (*)[10]'; did you mean to dereference it with '*'?}}

  NoBegin NB;
  for (auto i : NB) { }// expected-error{{range type 'NoBegin' has 'end' member but no 'begin' member}}
  NoBegin *pNB;
  for (auto i : pNB) { }// expected-error{{invalid range expression of type 'NoBegin *'; no viable 'begin' function available}}
  NoBegin **ppNB;
  for (auto i : ppNB) { }// expected-error{{invalid range expression of type 'NoBegin **'; no viable 'begin' function available}}
  NoBegin *****pppppNB;
  for (auto i : pppppNB) { }// expected-error{{invalid range expression of type 'NoBegin *****'; no viable 'begin' function available}}

  ADLNoEnd ANE;
  for (auto i : ANE) { } // expected-error{{invalid range expression of type 'ADLNoEnd'; no viable 'end' function available}}
  ADLNoEnd *pANE;
  for (auto i : pANE) { } // expected-error{{invalid range expression of type 'ADLNoEnd *'; no viable 'begin' function available}}

  DeletedEnd DE;
  for (auto i : DE) { } // expected-error{{attempt to use a deleted function}} \
expected-note {{when looking up 'end' function for range expression of type 'DeletedEnd'}}
  DeletedEnd *pDE;

  for (auto i : pDE) { } // expected-error {{invalid range expression of type 'DeletedEnd *'; no viable 'begin' function available}}

  PrivateEnd PE;
  // FIXME: This diagnostic should be improved, as it does not specify that
  // the range is invalid.
  for (auto i : PE) { } // expected-error{{'end' is a private member of 'PrivateEnd'}}

  PrivateEnd *pPE;
  for (auto i : pPE) { }// expected-error {{invalid range expression of type 'PrivateEnd *'}}
  // expected-error@-1 {{'end' is a private member of 'PrivateEnd'}}

  DeletedADLBegin DAB;
  for (auto i : DAB) { } // expected-error {{call to deleted function 'begin'}}\
  expected-note {{when looking up 'begin' function for range expression of type 'DeletedADLBegin'}}

  OverloadedStar OS;
  for (auto i : *OS) { }

  for (auto i : OS) { } // expected-error {{invalid range expression of type 'OverloadedStar'; did you mean to dereference it with '*'?}}

  for (Data *p : pt) { } // expected-error {{invalid range expression of type 'T *'; did you mean to dereference it with '*'?}}
  // expected-error@-1 {{no viable conversion from 'Data' to 'Data *'}}
  // expected-note@4 {{selected 'begin' function with iterator type 'Data *'}}
}
