// RUN: %clang_cc1 -fsyntax-only %s -verify
// rdar://13081751

typedef __SIZE_TYPE__ size_t;
void *memset(void*, int, size_t);

typedef struct __incomplete *incomplete;

void mt_query_for_domain(const char *domain)
{
	incomplete	query = 0;
	memset(query, 0, sizeof(query)); // expected-warning {{'memset' call operates on objects of type 'struct __incomplete' while the size is based on a different type 'incomplete'}} \
	// expected-note {{did you mean to dereference the argument to 'sizeof' (and multiply it by the number of elements)?}}
}

