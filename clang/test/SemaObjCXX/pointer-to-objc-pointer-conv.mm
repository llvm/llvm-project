// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface G
@end

@interface F
- (void)bar:(id *)objects;
- (void)foo:(G**)objects;
@end


void a() {
	F *b;
	G **keys;
	[b bar:keys];

	id *PID;
	[b foo:PID];

}


// pr7936
@interface I1 @end

class Wrapper {
public:
  operator id() const { return (id)_value; }
  operator Class() const { return (Class)_value; }
  operator I1*() const { return (I1*)_value; }

  bool Compare(id obj) { return *this == obj; }
  bool CompareClass(Class obj) { return *this == obj; }
  bool CompareI1(I1* obj) { return *this == obj; }

  Wrapper &operator*();
  Wrapper &operator[](int);
  Wrapper& operator->*(int);

private:
  long _value;
};

void f() {
  Wrapper w;
  w[0];
  *w;
  w->*(0);
}
