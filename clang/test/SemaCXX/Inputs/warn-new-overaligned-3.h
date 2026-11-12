#pragma GCC system_header

// This header file pretends to be <new> from the system library, for the
// purpose of the over-aligned warnings test.

void* operator new(unsigned long) {
  return 0;
}
void* operator new[](unsigned long) {
  return 0;
}

void* operator new(unsigned long, void *) {
  return 0;
}

void* operator new[](unsigned long, void *) {
  return 0;
}
