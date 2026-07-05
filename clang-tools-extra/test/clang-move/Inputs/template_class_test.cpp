#include "template_class_test.h"

template <typename T>
void A<T>::g() {}

template <typename T>
template <typename U>
void A<T>::k() {}

template <typename T>
int A<T>::c = 2;

void B::f() {}
