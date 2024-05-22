// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics
// PR9406

typedef struct {
	char *str;
	char *str2;
} Class;

typedef union {
	Class *object;
} Instance __attribute__((transparent_union));

__attribute__((overloadable)) void Class_Init(Instance this, char *str, void *str2) {
	this.object->str  = str;
	this.object->str2 = str2;
}

__attribute__((overloadable)) void Class_Init(Instance this, char *str) {
	this.object->str  = str;
	this.object->str2 = str;
}

int main(void) {
	Class obj;
	Class_Init(&obj, "Hello ", " World");
}

