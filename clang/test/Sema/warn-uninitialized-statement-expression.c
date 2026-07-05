// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -verify %s

void init(int *);

void foo(void) {
  int i = ({
    init(&i);
    i;
  });
}

void foo_bad(void) {
  int i = ({
    int z = i; // expected-warning{{variable 'i' is uninitialized when used within its own initialization}}
    init(&i);
    i;
  });
}

struct widget {
  int x, y;
};
void init2(struct widget *);

void bar(void) {
  struct widget my_widget = ({
    init2(&my_widget);
    my_widget;
  });
  struct widget a = (init2(&a), a);
}

void bar_bad(void) {
  struct widget my_widget = ({
    struct widget z = my_widget; // expected-warning{{variable 'my_widget' is uninitialized when used within its own initialization}}
    int x = my_widget.x;         //FIXME: There should be an uninitialized warning here
    init2(&my_widget);
    my_widget;
  });
}

void baz(void) {
  struct widget a = ({
    struct widget b = ({
      b = a; // expected-warning{{variable 'a' is uninitialized when used within its own initialization}}
    });
    a;
  });
}

void f(void) {
  struct widget *a = ({
    init2(a); // expected-warning{{variable 'a' is uninitialized when used within its own initialization}}
    a;
  });
}
