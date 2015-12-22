.. title:: clang-tidy - misc-inefficient-algorithm

misc-inefficient-algorithm
==========================


Warns on inefficient use of STL algorithms on associative containers.

Associative containers implements some of the algorithms as methods which
should be preferred to the algorithms in the algorithm header. The methods
can take advanatage of the order of the elements.
