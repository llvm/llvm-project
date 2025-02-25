#ifndef BASE_H
#define BASE_H

namespace NSP {
  struct CBase {
    unsigned B = 1;
    virtual void zero();
    virtual int one();
    virtual int two();
    virtual int three();
  };
}

extern void fooBase();
#endif
