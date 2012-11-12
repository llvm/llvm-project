// RUN: %clang_cc1 -g -S -o %t %s
// PR: 6554
// More then one anonymous aggregates on one line creates chaos when MDNode uniquness is 
// combined with RAUW operation.
// This test case causes crashes if malloc is configured to trip buffer overruns.
class MO {

  union {       struct {       union {    int BA;       } Val;       int Offset;     } OffsetedInfo;   } Contents; 

};

class MO m;
