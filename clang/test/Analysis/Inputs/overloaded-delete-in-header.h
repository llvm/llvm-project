#ifndef OVERLOADED_DELETE_IN_HEADER
#define OVERLOADED_DELETE_IN_HEADER

struct DeleteInHeader {
  int data;
  static void operator delete(void *ptr);
};

void DeleteInHeader::operator delete(void *ptr) {
  DeleteInHeader *self = (DeleteInHeader *)ptr;
  self->data = 1; // no-warning: Still alive.

  ::operator delete(ptr);

  self->data = 2; // expected-warning {{Use of memory after it is freed [cplusplus.NewDelete]}}
}

#endif // OVERLOADED_DELETE_IN_SYSTEM_HEADER
