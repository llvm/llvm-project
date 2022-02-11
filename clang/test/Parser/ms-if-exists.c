// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-extensions
// expected-no-diagnostics

struct Type {
};

void test_if_exists_stmts(void) {
  int b = 0;
  __if_exists(Type) {
    b++;
    b++;
  }
  __if_exists(Type_not) {
    this will not compile.
  }
  __if_not_exists(Type) {
    this will not compile.
  }
  __if_not_exists(Type_not) {
    b++;
    b++;
  }
}

int if_exists_creates_no_scope(void) {
  __if_exists(Type) {
    int x;  // 'x' is declared in the parent scope.
  }
  __if_not_exists(Type_not) {
    x++;
  }
  return x;
}

__if_exists(Type) {
  int var23;
}

__if_exists(Type_not) {
  this will not compile.
}

__if_not_exists(Type) {
  this will not compile.
}

__if_not_exists(Type_not) {
  int var244;
}

void test_if_exists_init_list(void) {

  int array1[] = {
    0,
    __if_exists(Type) {2, }
    3
  };

  int array2[] = {
    0,
    __if_exists(Type_not) { this will not compile }
    3
  };

  int array3[] = {
    0,
    __if_not_exists(Type_not) {2, }
    3
  };

  int array4[] = {
    0,
    __if_not_exists(Type) { this will not compile }
    3
  };

}


void test_nested_if_exists(void) {
  __if_exists(Type) {
    int x = 42;
    __if_not_exists(Type_not) {
      x++;
    }
  }
}
