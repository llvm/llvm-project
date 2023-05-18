// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

// Check that analysis-based warnings work in lambda bodies.
void analysis_based_warnings() {
  (void)[]() -> int { }; // expected-warning{{non-void lambda does not return a value}}
}

// Check that we get the right types of captured variables (the
// semantic-analysis part of p7).
int &check_const_int(int&);
float &check_const_int(const int&);

void test_capture_constness(int i, const int ic) {
  (void)[i,ic] ()->void {
    float &fr1 = check_const_int(i);
    float &fr2 = check_const_int(ic);
  }; 

  (void)[=] ()->void {
    float &fr1 = check_const_int(i);
    float &fr2 = check_const_int(ic);
  }; 

  (void)[i,ic] () mutable ->void {
    int &ir = check_const_int(i);
    float &fr = check_const_int(ic);
  };

  (void)[=] () mutable ->void {
    int &ir = check_const_int(i);
    float &fr = check_const_int(ic);
  };

  (void)[&i,&ic] ()->void {
    int &ir = check_const_int(i);
    float &fr = check_const_int(ic);
  };

  (void)[&] ()->void {
    int &ir = check_const_int(i);
    float &fr = check_const_int(ic);
  };
}


struct S1 {
  int x, y;
  S1 &operator=(int*);
  int operator()(int);
  void f() {
    [&]()->int {
      S1 &s1 = operator=(&this->x);
      return operator()(this->x + y);
    }(); 
  }
};
