// RUN: %clang_cc1 -fsyntax-only %s
// rdar://9584012

typedef struct {
	char *str;
} Class;

typedef union {
	Class *object;
} Instance __attribute__((transparent_union));

__attribute__((nonnull(1))) void Class_init(Instance this, char *str) {
	this.object->str = str;
}

int main(void) {
	Class *obj;
	Class_init(0, "Hello World"); // expected-warning {{null passed to a callee which requires a non-null argument}}
	Class_init(obj, "Hello World");
}

