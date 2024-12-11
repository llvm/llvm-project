#ifndef MOCK_HEADER_H
#define MOCK_HEADER_H

#pragma clang system_header
void *my_calloc(int count, int size) __attribute__((alloc_size(1,2)));

#endif /* MOCK_HEADER_H */
