// Header for PCH test cxx-typeid.cpp

#ifndef CXX_TYPEID_H
#define CXX_TYPEID_H

namespace std {

class type_info
{
public:
    virtual ~type_info();

    bool operator==(const type_info& rhs) const;
    bool operator!=(const type_info& rhs) const;

    bool before(const type_info& rhs) const;
    unsigned long hash_code() const;
    const char* name() const;

    type_info(const type_info& rhs);
    type_info& operator=(const type_info& rhs);
};

class bad_cast
{
public:
    bad_cast();
    bad_cast(const bad_cast&);
    bad_cast& operator=(const bad_cast&);
    virtual const char* what() const;
};

class bad_typeid
{
public:
    bad_typeid();
    bad_typeid(const bad_typeid&);
    bad_typeid& operator=(const bad_typeid&);
    virtual const char* what() const;
};

}  // std

#endif
