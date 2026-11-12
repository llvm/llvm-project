// RUN: %clang_cc1 %s -std=c++11 -fsyntax-only -Wmicrosoft -verify -fms-extensions

class MayExist {
private:
  typedef int Type;
};

void test_if_exists_stmts() {
  int b = 0;
  __if_exists(MayExist::Type) {
    b++;
    b++;
  }
  __if_exists(MayExist::Type_not) {
    this will not compile.
  }
  __if_not_exists(MayExist::Type) {
    this will not compile.
  }
  __if_not_exists(MayExist::Type_not) {
    b++;
    b++;
  }
}

int if_exists_creates_no_scope() {
  __if_exists(MayExist::Type) {
    int x;  // 'x' is declared in the parent scope.
  }
  __if_not_exists(MayExist::Type_not) {
    x++;
  }
  return x;
}

__if_exists(MayExist::Type) {
  int var23;
}

__if_exists(MayExist::Type_not) {
  this will not compile.
}

__if_not_exists(MayExist::Type) {
  this will not compile.
}

__if_not_exists(MayExist::Type_not) {
  int var244;
}

void test_if_exists_init_list() {

  int array1[] = {
    0,
    __if_exists(MayExist::Type) {2, }
    3
  };

  int array2[] = {
    0,
    __if_exists(MayExist::Type_not) { this will not compile }
    3
  };

  int array3[] = {
    0,
    __if_not_exists(MayExist::Type_not) {2, }
    3
  };

  int array4[] = {
    0,
    __if_not_exists(MayExist::Type) { this will not compile }
    3
  };

}


class IfExistsClassScope {
  __if_exists(MayExist::Type) {
    // __if_exists, __if_not_exists can nest
    __if_not_exists(MayExist::Type_not) {
      int var123;
    }
    int var23;
  }

  __if_exists(MayExist::Type_not) {
   this will not compile.
  }

  __if_not_exists(MayExist::Type) {
   this will not compile.
  }

  __if_not_exists(MayExist::Type_not) {
    int var244;
  }
};

void test_nested_if_exists() {
  __if_exists(MayExist::Type) {
    int x = 42;
    __if_not_exists(MayExist::Type_not) {
      x++;
    }
  }
}

void test_attribute_on_if_exists() {
  [[clang::fallthrough]] // expected-error {{an attribute list cannot appear here}}
  __if_exists(MayExist::Type) {
    int x;
  }
}
