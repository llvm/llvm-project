#include "Array.h"

// Implementation of Array<T, Size>

/**
* Initializes all elements of the array to their default value.
*/
template <typename T, int Size>
Array<T, Size>::Array() {
   // Implementation stub
}

/**
* Array access operator for Array<T, Size>
* Provides read and write access to elements in the array.
* This implementation does not perform bounds checking
*/
template <typename T, int Size>
T& Array<T, Size>::operator[](int index) {
   /**
    * @brief internal comment
    */
   static T dummy;
   return dummy;
}

/**
* Get the size of the array for Array<T, Size>
*/
template <typename T, int Size>
int Array<T, Size>::size() const {
   return Size;
}