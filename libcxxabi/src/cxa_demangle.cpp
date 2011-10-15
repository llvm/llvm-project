//===-------------------------- cxa_demangle.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "cxa_demangle.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <new>
#include <algorithm>
#include <assert.h>


#ifdef DEBUGGING

#include <string>
#include <typeinfo>

#endif

namespace __cxxabiv1
{

namespace __libcxxabi
{

#pragma GCC visibility push(hidden)

class __node
{
    __node(const __node&);
    __node& operator=(const __node&);
public:
    const char* __name_;
    size_t __size_;
    __node* __left_;
    __node* __right_;
    long double __value_;
    long __cached_size_;
public:
    __node()
        : __name_(0), __size_(0), __left_(0), __right_(0), __cached_size_(-1)
        {}
    virtual ~__node() {};

    void reset_cached_size()
    {
        __cached_size_ = -1;
        if (__left_)
            __left_->reset_cached_size();
        if (__right_)
            __right_->reset_cached_size();
    }

    virtual size_t first_size() const  {return 0;}
    virtual size_t second_size() const {return 0;}
    virtual size_t size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = first_size() + second_size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const {return buf;}
    virtual char* second_demangled_name(char* buf) const {return buf;}
    virtual char* get_demangled_name(char* buf) const
    {
        return second_demangled_name(first_demangled_name(buf));
    }
    virtual size_t base_size() const {return size();}
    virtual char* get_base_name(char* buf) const
    {
        return get_demangled_name(buf);
    }
    virtual ptrdiff_t print_base_name(char* f, char* l) const
    {
        return print(f, l);
    }
    virtual bool ends_with_template() const
    {
        return false;
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return false;
    }
    virtual __node* base_name() const
    {
        return const_cast<__node*>(this);
    }
    virtual bool is_reference_or_pointer_to_function_or_array() const
    {
        return false;
    }
    virtual bool is_function() const
    {
        return false;
    }
    virtual bool is_cv_qualifer() const
    {
        return false;
    }
    virtual bool is_array() const
    {
        return false;
    }

    virtual bool fix_forward_references(__node**, __node**)
    {
        return true;
    }
    virtual __node* extract_cv(__node*&) const
    {
        return 0;
    }
    virtual size_t list_len() const
    {
        return 0;
    }
    virtual bool is_sub() const
    {
        return false;
    }

    virtual ptrdiff_t print(char* f, char* l) const
    {
        const ptrdiff_t sz1 = print_first(f, l);
        return sz1 + print_second(f+std::min(sz1, l-f), l);
    }
    virtual ptrdiff_t print_first(char*, char*) const
    {
        return 0;
    }
    virtual ptrdiff_t print_second(char*, char*) const
    {
        return 0;
    }
};

#ifdef DEBUGGING

void display(__node* x, int indent = 0)
{
    if (x)
    {
        for (int i = 0; i < 2*indent; ++i)
            printf(" ");
        std::string buf(x->size(), '\0');
        x->print(&buf.front(), &buf.back()+1);
        printf("%s %s, %p\n", typeid(*x).name(), buf.c_str(), x);
        display(x->__left_, indent+1);
        display(x->__right_, indent+1);
    }
}

#endif

class __vtable
    : public __node
{
    static const ptrdiff_t n = sizeof("vtable for ") - 1;
public:
    __vtable(__node* type)
    {
        __right_ = type;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "vtable for ", n);
        return __right_->get_demangled_name(buf+n);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r < n)
            return n + __right_->print(l, l);
        ptrdiff_t sz = __right_->print(f+n, l) + n;
        if (r >= sz)
        {
            *f++ = 'v';
            *f++ = 't';
            *f++ = 'a';
            *f++ = 'b';
            *f++ = 'l';
            *f++ = 'e';
            *f++ = ' ';
            *f++ = 'f';
            *f++ = 'o';
            *f++ = 'r';
            *f   = ' ';
        }
        return sz;
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __VTT
    : public __node
{
    static const ptrdiff_t n = sizeof("VTT for ") - 1;
public:
    __VTT(__node* type)
    {
        __right_ = type;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "VTT for ", n);
        return __right_->get_demangled_name(buf+n);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r < n)
            return n + __right_->print(l, l);
        ptrdiff_t sz = __right_->print(f+n, l) + n;
        if (r >= sz)
        {
            *f++ = 'V';
            *f++ = 'T';
            *f++ = 'T';
            *f++ = ' ';
            *f++ = 'f';
            *f++ = 'o';
            *f++ = 'r';
            *f   = ' ';
        }
        return sz;
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __typeinfo
    : public __node
{
    static const ptrdiff_t n = sizeof("typeinfo for ") - 1;
public:
    __typeinfo(__node* type)
    {
        __right_ = type;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "typeinfo for ", n);
        return __right_->get_demangled_name(buf+n);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r < n)
            return n + __right_->print(l, l);
        ptrdiff_t sz = __right_->print(f+n, l) + n;
        if (r >= sz)
        {
            *f++ = 't';
            *f++ = 'y';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'i';
            *f++ = 'n';
            *f++ = 'f';
            *f++ = 'o';
            *f++ = ' ';
            *f++ = 'f';
            *f++ = 'o';
            *f++ = 'r';
            *f   = ' ';
        }
        return sz;
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __typeinfo_name
    : public __node
{
    static const ptrdiff_t n = sizeof("typeinfo name for ") - 1;
public:
    __typeinfo_name(__node* type)
    {
        __right_ = type;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "typeinfo name for ", n);
        return __right_->get_demangled_name(buf+n);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r < n)
            return n + __right_->print(l, l);
        ptrdiff_t sz = __right_->print(f+n, l) + n;
        if (r >= sz)
        {
            *f++ = 't';
            *f++ = 'y';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'i';
            *f++ = 'n';
            *f++ = 'f';
            *f++ = 'o';
            *f++ = ' ';
            *f++ = 'n';
            *f++ = 'a';
            *f++ = 'm';
            *f++ = 'e';
            *f++ = ' ';
            *f++ = 'f';
            *f++ = 'o';
            *f++ = 'r';
            *f   = ' ';
        }
        return sz;
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __covariant_return_thunk
    : public __node
{
    static const ptrdiff_t n = sizeof("covariant return thunk to ") - 1;
public:
    __covariant_return_thunk(__node* type)
    {
        __right_ = type;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "covariant return thunk to ", n);
        return __right_->get_demangled_name(buf+n);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r < n)
            return n + __right_->print(l, l);
        ptrdiff_t sz = __right_->print(f+n, l) + n;
        if (r >= sz)
        {
            *f++ = 'c';
            *f++ = 'o';
            *f++ = 'v';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = 'i';
            *f++ = 'a';
            *f++ = 'n';
            *f++ = 't';
            *f++ = ' ';
            *f++ = 'r';
            *f++ = 'e';
            *f++ = 't';
            *f++ = 'u';
            *f++ = 'r';
            *f++ = 'n';
            *f++ = ' ';
            *f++ = 't';
            *f++ = 'h';
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 'k';
            *f++ = ' ';
            *f++ = 't';
            *f++ = 'o';
            *f   = ' ';
        }
        return sz;
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __virtual_thunk
    : public __node
{
    static const size_t n = sizeof("virtual thunk to ") - 1;
public:
    __virtual_thunk(__node* type)
    {
        __right_ = type;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "virtual thunk to ", n);
        return __right_->get_demangled_name(buf+n);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r < n)
            return n + __right_->print(l, l);
        ptrdiff_t sz = __right_->print(f+n, l) + n;
        if (r >= sz)
        {
            *f++ = 'v';
            *f++ = 'i';
            *f++ = 'r';
            *f++ = 't';
            *f++ = 'u';
            *f++ = 'a';
            *f++ = 'l';
            *f++ = ' ';
            *f++ = 't';
            *f++ = 'h';
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 'k';
            *f++ = ' ';
            *f++ = 't';
            *f++ = 'o';
            *f   = ' ';
        }
        return sz;
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __non_virtual_thunk
    : public __node
{
    static const size_t n = sizeof("non-virtual thunk to ") - 1;
public:
    __non_virtual_thunk(__node* type)
    {
        __right_ = type;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "non-virtual thunk to ", n);
        return __right_->get_demangled_name(buf+n);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r < n)
            return n + __right_->print(l, l);
        ptrdiff_t sz = __right_->print(f+n, l) + n;
        if (r >= sz)
        {
            *f++ = 'n';
            *f++ = 'o';
            *f++ = 'n';
            *f++ = '-';
            *f++ = 'v';
            *f++ = 'i';
            *f++ = 'r';
            *f++ = 't';
            *f++ = 'u';
            *f++ = 'a';
            *f++ = 'l';
            *f++ = ' ';
            *f++ = 't';
            *f++ = 'h';
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 'k';
            *f++ = ' ';
            *f++ = 't';
            *f++ = 'o';
            *f   = ' ';
        }
        return sz;
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __guard_variable
    : public __node
{
    static const size_t n = sizeof("guard variable for ") - 1;
public:
    __guard_variable(__node* type)
    {
        __right_ = type;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "guard variable for ", n);
        return __right_->get_demangled_name(buf+n);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r < n)
            return n + __right_->print(l, l);
        ptrdiff_t sz = __right_->print(f+n, l) + n;
        if (r >= sz)
        {
            *f++ = 'g';
            *f++ = 'u';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 'v';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = 'i';
            *f++ = 'a';
            *f++ = 'b';
            *f++ = 'l';
            *f++ = 'e';
            *f++ = ' ';
            *f++ = 'f';
            *f++ = 'o';
            *f++ = 'r';
            *f   = ' ';
        }
        return sz;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __source_name
    : public __node
{
public:
    __source_name(const char* __name, unsigned __size)
    {
        __name_ = __name;
        __size_ = __size;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__size_ >= 10 && strncmp(__name_, "_GLOBAL__N", 10) == 0)
                const_cast<long&>(__cached_size_) = 21;
            else
                const_cast<long&>(__cached_size_) = __size_;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__size_ >= 10 && strncmp(__name_, "_GLOBAL__N", 10) == 0)
            return strncpy(buf, "(anonymous namespace)", 21) + 21;
        return strncpy(buf, __name_, __size_) + __size_;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__size_ >= 10 && strncmp(__name_, "_GLOBAL__N", 10) == 0)
        {
            const ptrdiff_t n = sizeof("(anonymous namespace)") - 1;
            if (r >= n)
            {
                *f++ = '(';
                *f++ = 'a';
                *f++ = 'n';
                *f++ = 'o';
                *f++ = 'n';
                *f++ = 'y';
                *f++ = 'm';
                *f++ = 'o';
                *f++ = 'u';
                *f++ = 's';
                *f++ = ' ';
                *f++ = 'n';
                *f++ = 'a';
                *f++ = 'm';
                *f++ = 'e';
                *f++ = 's';
                *f++ = 'p';
                *f++ = 'a';
                *f++ = 'c';
                *f++ = 'e';
                *f   = ')';
            }
            return n;
        }
        if (r >= __size_)
            strncpy(f, __name_, __size_);
        return __size_;
    }
};

class __operator_new
    : public __node
{
public:

    virtual size_t first_size() const {return sizeof("operator new") - 1;}
    virtual char* first_demangled_name(char* buf) const
    {
        return strncpy(buf, "operator new", sizeof("operator new") - 1) +
                                            sizeof("operator new") - 1;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("operator new") - 1;
        if (r >= n)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = ' ';
            *f++ = 'n';
            *f++ = 'e';
            *f   = 'w';
        }
        return n;
    }
};

class __operator_new_array
    : public __node
{
public:

    virtual size_t first_size() const {return sizeof("operator new[]") - 1;}
    virtual char* first_demangled_name(char* buf) const
    {
        return strncpy(buf, "operator new[]", sizeof("operator new[]") - 1) +
                                              sizeof("operator new[]") - 1;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("operator new[]") - 1;
        if (r >= n)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = ' ';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'w';
            *f++ = '[';
            *f   = ']';
        }
        return n;
    }
};

class __operator_delete
    : public __node
{
public:

    virtual size_t first_size() const {return sizeof("operator delete") - 1;}
    virtual char* first_demangled_name(char* buf) const
    {
        return strncpy(buf, "operator delete", sizeof("operator delete") - 1) +
                                               sizeof("operator delete") - 1;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("operator delete") - 1;
        if (r >= n)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = ' ';
            *f++ = 'd';
            *f++ = 'e';
            *f++ = 'l';
            *f++ = 'e';
            *f++ = 't';
            *f   = 'e';
        }
        return n;
    }
};

class __operator_delete_array
    : public __node
{
public:

    virtual size_t first_size() const {return sizeof("operator delete[]") - 1;}
    virtual char* first_demangled_name(char* buf) const
    {
        return strncpy(buf, "operator delete[]", sizeof("operator delete[]") - 1) +
                                                 sizeof("operator delete[]") - 1;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("operator delete[]") - 1;
        if (r >= n)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = ' ';
            *f++ = 'd';
            *f++ = 'e';
            *f++ = 'l';
            *f++ = 'e';
            *f++ = 't';
            *f++ = 'e';
            *f++ = '[';
            *f   = ']';
        }
        return n;
    }
};

class __operator_logical_and
    : public __node
{
public:

    __operator_logical_and() {}
    __operator_logical_and(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator&&") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") && (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator&&", sizeof("operator&&") - 1);
            buf += sizeof("operator&&") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '&';
                *f++ = '&';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator&&") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '&';
            *f   = '&';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_addressof
    : public __node
{
public:

    __operator_addressof() {}
    explicit __operator_addressof(__node* op)
    {
        __left_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = 3+__left_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator&") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '&';
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator&", sizeof("operator&") - 1);
            buf += sizeof("operator&") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 3;
            if (r < n1)
                return n1 + __left_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+2, l);
            if (r >= n1 + sz1)
            {
                *f++ = '&';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator&") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '&';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__left_)
            return __left_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_bit_and
    : public __node
{
public:

    __operator_bit_and() {}
    __operator_bit_and(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator&") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") & (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator&", sizeof("operator&") - 1);
            buf += sizeof("operator&") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '&';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator&") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '&';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_and_equal
    : public __node
{
public:

    __operator_and_equal() {}
    __operator_and_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator&=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") &= (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator&=", sizeof("operator&=") - 1);
            buf += sizeof("operator&=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '&';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator&=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '&';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_equal
    : public __node
{
public:

    __operator_equal() {}
    __operator_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") = (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator=", sizeof("operator=") - 1);
            buf += sizeof("operator=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_alignof_type
    : public __node
{
public:

    __operator_alignof_type() {}
    __operator_alignof_type(__node* op)
    {
        __right_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__right_)
                const_cast<long&>(__cached_size_) = __right_->size() + 10;
            else
                const_cast<long&>(__cached_size_) = sizeof("operator alignof") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__right_)
        {
            strncpy(buf, "alignof (", 9);
            buf += 9;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator alignof", sizeof("operator alignof") - 1);
            buf += sizeof("operator alignof") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__right_)
        {
            const ptrdiff_t n1 = sizeof("alignof ()") - 1;
            if (r < n1)
                return n1 + __right_->print(l, l);
            ptrdiff_t sz1 = __right_->print(f+(n1-1), l);
            if (r >= n1 + sz1)
            {
                *f++ = 'a';
                *f++ = 'l';
                *f++ = 'i';
                *f++ = 'g';
                *f++ = 'n';
                *f++ = 'o';
                *f++ = 'f';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator alignof") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = ' ';
            *f++ = 'a';
            *f++ = 'l';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'o';
            *f   = 'f';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__right_)
            return __right_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_alignof_expression
    : public __node
{
public:

    __operator_alignof_expression() {}
    __operator_alignof_expression(__node* op)
    {
        __right_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__right_)
                const_cast<long&>(__cached_size_) = __right_->size() + 10;
            else
                const_cast<long&>(__cached_size_) = sizeof("operator alignof") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__right_)
        {
            strncpy(buf, "alignof (", 9);
            buf += 9;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator alignof", sizeof("operator alignof") - 1);
            buf += sizeof("operator alignof") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__right_)
        {
            const ptrdiff_t n1 = sizeof("alignof ()") - 1;
            if (r < n1)
                return n1 + __right_->print(l, l);
            ptrdiff_t sz1 = __right_->print(f+(n1-1), l);
            if (r >= n1 + sz1)
            {
                *f++ = 'a';
                *f++ = 'l';
                *f++ = 'i';
                *f++ = 'g';
                *f++ = 'n';
                *f++ = 'o';
                *f++ = 'f';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator alignof") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = ' ';
            *f++ = 'a';
            *f++ = 'l';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'o';
            *f   = 'f';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__right_)
            return __right_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_paren
    : public __node
{
public:

    virtual size_t first_size() const {return sizeof("operator()") - 1;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "operator()", sizeof("operator()") - 1);
        return buf + sizeof("operator()") - 1;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("operator()") - 1;
        if (r >= n)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '(';
            *f   = ')';
        }
        return n;
    }
};

class __operator_comma
    : public __node
{
public:

    __operator_comma() {}
    __operator_comma(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator,") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") , (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator,", sizeof("operator,") - 1);
            buf += sizeof("operator,") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = ',';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator,") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = ',';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_tilda
    : public __node
{
public:

    __operator_tilda() {}
    explicit __operator_tilda(__node* op)
    {
        __left_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = 3+__left_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator~") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '~';
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator~", sizeof("operator~") - 1);
            buf += sizeof("operator~") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 3;
            if (r < n1)
                return n1 + __left_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+2, l);
            if (r >= n1 + sz1)
            {
                *f++ = '~';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator~") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '~';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__left_)
            return __left_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_cast
    : public __node
{
    static const size_t n = sizeof("operator ") - 1;
public:

    explicit __operator_cast(__node* type)
    {
        __right_ = type;
    }
    __operator_cast(__node* type, __node* arg)
    {
        __size_ = 1;
        __right_ = type;
        __left_ = arg;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            size_t off;
            if (__size_)
            {
                off = 4;
                off += __right_->size();
                if (__left_)
                    off += __left_->size();
            }
            else
                off = n +  __right_->size();;
            const_cast<long&>(__cached_size_) = off;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__size_)
        {
            *buf++ = '(';
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
            *buf++ = '(';
            if (__left_)
                buf = __left_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator ", n);
            buf = __right_->get_demangled_name(buf+n);
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__size_)
        {
            const ptrdiff_t n1 = 4;
            if (r < n1)
                return n1 + __right_->print(l, l) +
                            (__left_ ? __left_->print(l, l) : 0);
            ptrdiff_t sz1 = __right_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + (__left_ ? __left_->print(l, l) : 0);
            ptrdiff_t sz2 = __left_ ? __left_->print(f+3+sz1, l) : 0;
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator ") - 1;
        if (r < n2)
            return n2 + __right_->print(l, l);
        ptrdiff_t sz1 = __right_->print(f+n2, l);
        if (r >= n2 + sz1)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = ' ';
        }
        return n2 + sz1;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return true;
    }
};

class __cast_literal
    : public __node
{
public:

    __cast_literal(__node* type, const char* f, const char* l)
    {
        __left_ = type;
        __name_ = f;
        __size_ = l - f;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = 2 + __left_->size() + __size_;
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        *buf++ = '(';
        buf = __left_->get_demangled_name(buf);
        *buf++ = ')';
        strncpy(buf, __name_, __size_);
        return buf + __size_;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = 2;
        if (r < __size_ + n)
            return __size_ + n + __left_->print(l, l);
        ptrdiff_t sz = __left_->print(f+1, l);
        if (r >= __size_ + n + sz)
        {
            *f   = '(';
            f += 1 + sz;
            *f++ = ')';
            strncpy(f, __name_, __size_);
        }
        return __size_ + n + sz;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
};

class __operator_dereference
    : public __node
{
public:

    __operator_dereference() {}
    explicit __operator_dereference(__node* op)
    {
        __left_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = 3+__left_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator*") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '*';
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator*", sizeof("operator*") - 1);
            buf += sizeof("operator*") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 3;
            if (r < n1)
                return n1 + __left_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+2, l);
            if (r >= n1 + sz1)
            {
                *f++ = '*';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator*") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '*';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__left_)
            return __left_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_divide
    : public __node
{
public:

    __operator_divide() {}
    __operator_divide(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator/") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") / (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator/", sizeof("operator/") - 1);
            buf += sizeof("operator/") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '/';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator/") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '/';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_divide_equal
    : public __node
{
public:

    __operator_divide_equal() {}
    __operator_divide_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator/=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") /= (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator/=", sizeof("operator/=") - 1);
            buf += sizeof("operator/=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '/';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator/=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '/';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_xor
    : public __node
{
public:

    __operator_xor() {}
    __operator_xor(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator^") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") ^ (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator^", sizeof("operator^") - 1);
            buf += sizeof("operator^") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '^';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator^") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '^';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_xor_equal
    : public __node
{
public:

    __operator_xor_equal() {}
    __operator_xor_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator^=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';  // strncpy(buf, "(", 1);
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") ^= (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator^=", sizeof("operator^=") - 1);
            buf += sizeof("operator^=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '^';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator^=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '^';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_equality
    : public __node
{
public:

    __operator_equality() {}
    __operator_equality(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator==") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") == (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator==", sizeof("operator==") - 1);
            buf += sizeof("operator==") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '=';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator==") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '=';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_greater_equal
    : public __node
{
public:

    __operator_greater_equal() {}
    __operator_greater_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator>=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") >= (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator>=", sizeof("operator>=") - 1);
            buf += sizeof("operator>=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '>';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator>=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '>';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_greater
    : public __node
{
public:

    __operator_greater() {}
    __operator_greater(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 9 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator>") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") > (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator>", sizeof("operator>") - 1);
            buf += sizeof("operator>") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 9;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+2, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-2)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f++ = '(';
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '>';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f++ = ')';
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator>") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '>';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_brackets
    : public __node
{
public:

    virtual size_t first_size() const {return sizeof("operator[]") - 1;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "operator[]", sizeof("operator[]") - 1);
        return buf + sizeof("operator[]") - 1;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("operator[]") - 1;
        if (r >= n)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '[';
            *f   = ']';
        }
        return n;
    }
};

class __operator_less_equal
    : public __node
{
public:

    __operator_less_equal() {}
    __operator_less_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator<=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") <= (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator<=", sizeof("operator<=") - 1);
            buf += sizeof("operator<=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '<';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator<=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '<';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_less
    : public __node
{
public:

    __operator_less() {}
    __operator_less(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator<") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") < (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator<", sizeof("operator<") - 1);
            buf += sizeof("operator<") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '<';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator<") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '<';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_left_shift
    : public __node
{
public:

    __operator_left_shift() {}
    __operator_left_shift(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator<<") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") << (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator<<", sizeof("operator<<") - 1);
            buf += sizeof("operator<<") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '<';
                *f++ = '<';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator<<") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '<';
            *f   = '<';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_left_shift_equal
    : public __node
{
public:

    __operator_left_shift_equal() {}
    __operator_left_shift_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 9 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator<<=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") <<= (", 7);
            buf += 7;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator<<=", sizeof("operator<<=") - 1);
            buf += sizeof("operator<<=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 9;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '<';
                *f++ = '<';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator<<=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '<';
            *f++ = '<';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_minus
    : public __node
{
public:

    __operator_minus() {}
    __operator_minus(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator-") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") - (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator-", sizeof("operator-") - 1);
            buf += sizeof("operator-") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '-';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator-") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '-';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_minus_equal
    : public __node
{
public:

    __operator_minus_equal() {}
    __operator_minus_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator-=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") -= (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator-=", sizeof("operator-=") - 1);
            buf += sizeof("operator-=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '-';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator-=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '-';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_times
    : public __node
{
public:

    __operator_times() {}
    __operator_times(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator*") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") * (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator*", sizeof("operator*") - 1);
            buf += sizeof("operator*") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '*';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator*") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '*';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_times_equal
    : public __node
{
public:

    __operator_times_equal() {}
    __operator_times_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator*=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") *= (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator*=", sizeof("operator*=") - 1);
            buf += sizeof("operator*=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '*';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator*=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '*';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_decrement
    : public __node
{
public:

    __operator_decrement() {}
    explicit __operator_decrement(bool prefix, __node* op)
    {
        __size_ = prefix;
        __left_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = 4+__left_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator--") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            if (__size_)
            {
                *buf++ = '-';
                *buf++ = '-';
                *buf++ = '(';
            }
            else
                *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            if (__size_)
                *buf++ = ')';
            else
            {
                *buf++ = ')';
                *buf++ = '-';
                *buf++ = '-';
            }
        }
        else
        {
            strncpy(buf, "operator--", sizeof("operator--") - 1);
            buf += sizeof("operator--") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 4;
            if (r < n1)
                return n1 + __left_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f + (__size_ ? 3 : 1), l);
            if (r >= n1 + sz1)
            {
                if (__size_)
                {
                    *f++ = '-';
                    *f++ = '-';
                    *f   = '(';
                    f += 1+sz1;
                    *f   = ')';
                }
                else
                {
                    *f   = '(';
                    f += 1+sz1;
                    *f++ = ')';
                    *f++ = '-';
                    *f   = '-';
                }
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator--") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '-';
            *f   = '-';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__left_)
            return __left_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_not_equal
    : public __node
{
public:

    __operator_not_equal() {}
    __operator_not_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator!=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") != (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator!=", sizeof("operator!=") - 1);
            buf += sizeof("operator!=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '!';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator!=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '!';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_negate
    : public __node
{
public:

    __operator_negate() {}
    explicit __operator_negate(__node* op)
    {
        __left_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = 3+__left_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator-") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '-';
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator-", sizeof("operator-") - 1);
            buf += sizeof("operator-") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 3;
            if (r < n1)
                return n1 + __left_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+2, l);
            if (r >= n1 + sz1)
            {
                *f++ = '-';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator-") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '-';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__left_)
            return __left_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_logical_not
    : public __node
{
public:

    __operator_logical_not() {}
    explicit __operator_logical_not(__node* op)
    {
        __left_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = 3+__left_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator!") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '!';
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator!", sizeof("operator!") - 1);
            buf += sizeof("operator!") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 3;
            if (r < n1)
                return n1 + __left_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+2, l);
            if (r >= n1 + sz1)
            {
                *f++ = '!';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator!") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '!';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__left_)
            return __left_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_logical_or
    : public __node
{
public:

    __operator_logical_or() {}
    __operator_logical_or(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator||") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") || (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator||", sizeof("operator||") - 1);
            buf += sizeof("operator||") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '|';
                *f++ = '|';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator||") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '|';
            *f   = '|';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_bit_or
    : public __node
{
public:

    __operator_bit_or() {}
    __operator_bit_or(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator|") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") | (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator|", sizeof("operator|") - 1);
            buf += sizeof("operator|") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '|';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator|") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '|';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_or_equal
    : public __node
{
public:

    __operator_or_equal() {}
    __operator_or_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator|=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") |= (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator|=", sizeof("operator|=") - 1);
            buf += sizeof("operator|=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '|';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator|=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '|';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_pointer_to_member
    : public __node
{
public:

    __operator_pointer_to_member() {}
    __operator_pointer_to_member(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 9 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator->*") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") ->* (", 7);
            buf += 7;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator->*", sizeof("operator->*") - 1);
            buf += sizeof("operator->*") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 9;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '-';
                *f++ = '>';
                *f++ = '*';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator->*") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '-';
            *f++ = '>';
            *f   = '*';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_plus
    : public __node
{
public:

    __operator_plus() {}
    __operator_plus(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator+") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") + (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator+", sizeof("operator+") - 1);
            buf += sizeof("operator+") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '+';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator+") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '+';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_plus_equal
    : public __node
{
public:

    __operator_plus_equal() {}
    __operator_plus_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator+=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") += (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator+=", sizeof("operator+=") - 1);
            buf += sizeof("operator+=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '+';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator+=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '+';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_increment
    : public __node
{
public:

    __operator_increment() {}
    explicit __operator_increment(bool prefix, __node* op)
    {
        __size_ = prefix;
        __left_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = 4+__left_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator++") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            if (__size_)
            {
                *buf++ = '+';
                *buf++ = '+';
                *buf++ = '(';
            }
            else
                *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            if (__size_)
                *buf++ = ')';
            else
            {
                *buf++ = ')';
                *buf++ = '+';
                *buf++ = '+';
            }
        }
        else
        {
            strncpy(buf, "operator++", sizeof("operator++") - 1);
            buf += sizeof("operator++") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 4;
            if (r < n1)
                return n1 + __left_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f + (__size_ ? 3 : 1), l);
            if (r >= n1 + sz1)
            {
                if (__size_)
                {
                    *f++ = '+';
                    *f++ = '+';
                    *f   = '(';
                    f += 1+sz1;
                    *f   = ')';
                }
                else
                {
                    *f   = '(';
                    f += 1+sz1;
                    *f++ = ')';
                    *f++ = '+';
                    *f   = '+';
                }
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator++") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '+';
            *f   = '+';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__left_)
            return __left_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_unary_plus
    : public __node
{
public:

    __operator_unary_plus() {}
    explicit __operator_unary_plus(__node* op)
    {
        __left_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = 3+__left_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator+") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '+';
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator+", sizeof("operator+") - 1);
            buf += sizeof("operator+") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 3;
            if (r < n1)
                return n1 + __left_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+2, l);
            if (r >= n1 + sz1)
            {
                *f++ = '+';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator+") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '+';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__left_)
            return __left_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_arrow
    : public __node
{
public:

    __operator_arrow() {}
    __operator_arrow(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator->") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") -> (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator->", sizeof("operator->") - 1);
            buf += sizeof("operator->") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '-';
                *f++ = '>';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator->") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '-';
            *f   = '>';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_conditional
    : public __node
{
public:

    __operator_conditional() {}
    __operator_conditional(__node* op1, __node* op2, __node* op3)
    {
        __name_ = (const char*)op1;
        __left_ = op2;
        __right_ = op3;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
            {
                __node* op1 = (__node*)__name_;
                const_cast<long&>(__cached_size_) = op1->size() + __left_->size() + 12 + __right_->size();
            }
            else
                const_cast<long&>(__cached_size_) = sizeof("operator?") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            __node* op1 = (__node*)__name_;
            *buf++ = '(';
            buf = op1->get_demangled_name(buf);
            strncpy(buf, ") ? (", 5);
            buf += 5;
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") : (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator?", sizeof("operator?") - 1);
            buf += sizeof("operator?") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 12;
            __node* op1 = (__node*)__name_;
            if (r < n1)
                return n1 + op1->print(l, l) + __left_->print(l, l) +
                                               __right_->print(l, l);
            ptrdiff_t sz1 = op1->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz2 = __left_->print(f+6+sz1, l);
            if (r < n1 + sz1 + sz2)
                return n1 + sz1 + sz2 + __right_->print(l, l);
            ptrdiff_t sz3 = __right_->print(f+11+sz1+sz2, l);
            if (r >= n1 + sz1 + sz2 + sz3)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '?';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f++ = ')';
                *f++ = ' ';
                *f++ = ':';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz3;
                *f   = ')';
            }
            return n1 + sz1 + sz2 + sz3;
        }
        const ptrdiff_t n2 = sizeof("operator?") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '?';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__name_)
            r = r && ((__node*)__name_)->fix_forward_references(t_begin, t_end);
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_mod
    : public __node
{
public:

    __operator_mod() {}
    __operator_mod(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 7 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator%") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") % (", 5);
            buf += 5;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator%", sizeof("operator%") - 1);
            buf += sizeof("operator%") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 7;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '%';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator%") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f   = '%';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_mod_equal
    : public __node
{
public:

    __operator_mod_equal() {}
    __operator_mod_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator%=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") %= (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator%=", sizeof("operator%=") - 1);
            buf += sizeof("operator%=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '%';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator%=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '%';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_right_shift
    : public __node
{
public:

    __operator_right_shift() {}
    __operator_right_shift(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 8 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator>>") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") >> (", 6);
            buf += 6;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator>>", sizeof("operator>>") - 1);
            buf += sizeof("operator>>") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 8;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '>';
                *f++ = '>';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator>>") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '>';
            *f   = '>';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_right_shift_equal
    : public __node
{
public:

    __operator_right_shift_equal() {}
    __operator_right_shift_equal(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_)
                const_cast<long&>(__cached_size_) = __left_->size() + 9 + __right_->size();
            else
                const_cast<long&>(__cached_size_) = sizeof("operator>>=") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            strncpy(buf, ") >>= (", 7);
            buf += 7;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator>>=", sizeof("operator>>=") - 1);
            buf += sizeof("operator>>=") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__left_)
        {
            const ptrdiff_t n1 = 9;
            if (r < n1)
                return n1 + __left_->print(l, l) + __right_->print(l, l);
            ptrdiff_t sz1 = __left_->print(f+1, l);
            if (r < n1 + sz1)
                return n1 + sz1 + __right_->print(l, l);
            ptrdiff_t sz2 = __right_->print(f+(n1-1)+sz1, l);
            if (r >= n1 + sz1 + sz2)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
                *f++ = ' ';
                *f++ = '>';
                *f++ = '>';
                *f++ = '=';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz2;
                *f   = ')';
            }
            return n1 + sz1 + sz2;
        }
        const ptrdiff_t n2 = sizeof("operator>>=") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = '>';
            *f++ = '>';
            *f   = '=';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __operator_sizeof_type
    : public __node
{
public:

    __operator_sizeof_type() {}
    __operator_sizeof_type(__node* op)
    {
        __right_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__right_)
                const_cast<long&>(__cached_size_) = __right_->size() + 9;
            else
                const_cast<long&>(__cached_size_) = sizeof("operator sizeof") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__right_)
        {
            strncpy(buf, "sizeof (", 8);
            buf += 8;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator sizeof", sizeof("operator sizeof") - 1);
            buf += sizeof("operator sizeof") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__right_)
        {
            const ptrdiff_t n1 = sizeof("sizeof ()") - 1;
            if (r < n1)
                return n1 + __right_->print(l, l);
            ptrdiff_t sz1 = __right_->print(f+(n1-1), l);
            if (r >= n1 + sz1)
            {
                *f++ = 's';
                *f++ = 'i';
                *f++ = 'z';
                *f++ = 'e';
                *f++ = 'o';
                *f++ = 'f';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator sizeof") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = ' ';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'z';
            *f++ = 'e';
            *f++ = 'o';
            *f   = 'f';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__right_)
            return __right_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __operator_sizeof_expression
    : public __node
{
public:

    __operator_sizeof_expression() {}
    __operator_sizeof_expression(__node* op)
    {
        __right_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__right_)
                const_cast<long&>(__cached_size_) = __right_->size() + 9;
            else
                const_cast<long&>(__cached_size_) = sizeof("operator sizeof") - 1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__right_)
        {
            strncpy(buf, "sizeof (", 8);
            buf += 8;
            buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        else
        {
            strncpy(buf, "operator sizeof", sizeof("operator sizeof") - 1);
            buf += sizeof("operator sizeof") - 1;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__right_)
        {
            const ptrdiff_t n1 = sizeof("sizeof ()") - 1;
            if (r < n1)
                return n1 + __right_->print(l, l);
            ptrdiff_t sz1 = __right_->print(f+(n1-1), l);
            if (r >= n1 + sz1)
            {
                *f++ = 's';
                *f++ = 'i';
                *f++ = 'z';
                *f++ = 'e';
                *f++ = 'o';
                *f++ = 'f';
                *f++ = ' ';
                *f   = '(';
                f += 1 + sz1;
                *f   = ')';
            }
            return n1 + sz1;
        }
        const ptrdiff_t n2 = sizeof("operator sizeof") - 1;
        if (r >= n2)
        {
            *f++ = 'o';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = ' ';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'z';
            *f++ = 'e';
            *f++ = 'o';
            *f   = 'f';
        }
        return n2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__right_)
            return __right_->fix_forward_references(t_begin, t_end);
        return true;
    }
};

class __typeid
    : public __node
{
public:

    __typeid(__node* op)
    {
        __right_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __right_->size() + 8;
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "typeid(", 7);
        buf += 7;
        buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("typeid()") - 1;
            if (r < n1)
                return n1 + __right_->print(l, l);
        ptrdiff_t sz1 = __right_->print(f+(n1-1), l);
        if (r >= n1 + sz1)
        {
            *f++ = 't';
            *f++ = 'y';
            *f++ = 'p';
            *f++ = 'e';
            *f++ = 'i';
            *f++ = 'd';
            *f   = '(';
            f += 1 + sz1;
            *f   = ')';
        }
        return n1 + sz1;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __throw
    : public __node
{
public:

    __throw(__node* op)
    {
        __right_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __right_->size() + 6;
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "throw ", 6);
        return __right_->get_demangled_name(buf+6);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("throw ") - 1;
        if (r < n1)
            return n1 + __right_->print(l, l);
        ptrdiff_t sz1 = __right_->print(f+n1, l);
        if (r >= n1 + sz1)
        {
            *f++ = 't';
            *f++ = 'h';
            *f++ = 'r';
            *f++ = 'o';
            *f++ = 'w';
            *f   = ' ';
        }
        return n1 + sz1;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __rethrow
    : public __node
{
    static const ptrdiff_t n = sizeof("throw") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "throw", n);
        return buf+n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 't';
            *f++ = 'h';
            *f++ = 'r';
            *f++ = 'o';
            *f   = 'w';
        }
        return n;
    }
};

class __operator_sizeof_param_pack
    : public __node
{
public:

    __operator_sizeof_param_pack(__node* op)
    {
        __right_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __right_->size() + 11;
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "sizeof...(", 10);
        buf += 10;
        buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("sizeof...()") - 1;
        if (r < n1)
            return n1 + __right_->print(l, l);
        ptrdiff_t sz1 = __right_->print(f+(n1-1), l);
        if (r >= n1 + sz1)
        {
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'z';
            *f++ = 'e';
            *f++ = 'o';
            *f++ = 'f';
            *f++ = '.';
            *f++ = '.';
            *f++ = '.';
            *f   = '(';
            f += 1+sz1;
            *f   = ')';
        }
        return n1 + sz1;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __const_cast
    : public __node
{
public:

    __const_cast(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + 14 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "const_cast<", 11);
        buf += 11;
        buf = __left_->get_demangled_name(buf);
        *buf++ = '>';
        *buf++ = '(';
        buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("const_cast<>()") - 1;
        if (r < n1)
            return n1 + __left_->print(l, l) + __right_->print(l, l);
        ptrdiff_t sz1 = __left_->print(f+(n1-3), l);
        if (r < n1 + sz1)
            return n1 + sz1 + __right_->print(l, l);
        ptrdiff_t sz2 = __right_->print(f+sz1+(n1-1), l);
        if (r >= n1 + sz1 + sz2)
        {
            *f++ = 'c';
            *f++ = 'o';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 't';
            *f++ = '_';
            *f++ = 'c';
            *f++ = 'a';
            *f++ = 's';
            *f++ = 't';
            *f   = '<';
            f += 1+sz1;
            *f++ = '>';
            *f   = '(';
            f += 1+sz2;
            *f   = ')';
        }
        return n1 + sz1 + sz2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) &&
              __right_->fix_forward_references(t_begin, t_end);
    }
};

class __dynamic_cast
    : public __node
{
public:

    __dynamic_cast(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + 16 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "dynamic_cast<", 13);
        buf += 13;
        buf = __left_->get_demangled_name(buf);
        *buf++ = '>';
        *buf++ = '(';
        buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("dynamic_cast<>()") - 1;
        if (r < n1)
            return n1 + __left_->print(l, l) + __right_->print(l, l);
        ptrdiff_t sz1 = __left_->print(f+(n1-3), l);
        if (r < n1 + sz1)
            return n1 + sz1 + __right_->print(l, l);
        ptrdiff_t sz2 = __right_->print(f+sz1+(n1-1), l);
        if (r >= n1 + sz1 + sz2)
        {
            *f++ = 'd';
            *f++ = 'y';
            *f++ = 'n';
            *f++ = 'a';
            *f++ = 'm';
            *f++ = 'i';
            *f++ = 'c';
            *f++ = '_';
            *f++ = 'c';
            *f++ = 'a';
            *f++ = 's';
            *f++ = 't';
            *f   = '<';
            f += 1+sz1;
            *f++ = '>';
            *f   = '(';
            f += 1+sz2;
            *f   = ')';
        }
        return n1 + sz1 + sz2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) &&
              __right_->fix_forward_references(t_begin, t_end);
    }
};

class __reinterpret_cast
    : public __node
{
public:

    __reinterpret_cast(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + 20 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "reinterpret_cast<", 17);
        buf += 17;
        buf = __left_->get_demangled_name(buf);
        *buf++ = '>';
        *buf++ = '(';
        buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("reinterpret_cast<>()") - 1;
        if (r < n1)
            return n1 + __left_->print(l, l) + __right_->print(l, l);
        ptrdiff_t sz1 = __left_->print(f+(n1-3), l);
        if (r < n1 + sz1)
            return n1 + sz1 + __right_->print(l, l);
        ptrdiff_t sz2 = __right_->print(f+sz1+(n1-1), l);
        if (r >= n1 + sz1 + sz2)
        {
            *f++ = 'r';
            *f++ = 'e';
            *f++ = 'i';
            *f++ = 'n';
            *f++ = 't';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'p';
            *f++ = 'r';
            *f++ = 'e';
            *f++ = 't';
            *f++ = '_';
            *f++ = 'c';
            *f++ = 'a';
            *f++ = 's';
            *f++ = 't';
            *f   = '<';
            f += 1+sz1;
            *f++ = '>';
            *f   = '(';
            f += 1+sz2;
            *f   = ')';
        }
        return n1 + sz1 + sz2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) &&
              __right_->fix_forward_references(t_begin, t_end);
    }
};

class __static_cast
    : public __node
{
public:

    __static_cast(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + 15 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "static_cast<", 12);
        buf += 12;
        buf = __left_->get_demangled_name(buf);
        *buf++ = '>';
        *buf++ = '(';
        buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("static_cast<>()") - 1;
        if (r < n1)
            return n1 + __left_->print(l, l) + __right_->print(l, l);
        ptrdiff_t sz1 = __left_->print(f+(n1-3), l);
        if (r < n1 + sz1)
            return n1 + sz1 + __right_->print(l, l);
        ptrdiff_t sz2 = __right_->print(f+sz1+(n1-1), l);
        if (r >= n1 + sz1 + sz2)
        {
            *f++ = 's';
            *f++ = 't';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'i';
            *f++ = 'c';
            *f++ = '_';
            *f++ = 'c';
            *f++ = 'a';
            *f++ = 's';
            *f++ = 't';
            *f   = '<';
            f += 1+sz1;
            *f++ = '>';
            *f   = '(';
            f += 1+sz2;
            *f   = ')';
        }
        return n1 + sz1 + sz2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) &&
              __right_->fix_forward_references(t_begin, t_end);
    }
};

class __call_expr
    : public __node
{
public:

    __call_expr(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            size_t off = __left_->size() + 2;
            if (__right_)
                off += __right_->size();
            const_cast<long&>(__cached_size_) = off;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        *buf++ = '(';
        if (__right_)
            buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("()") - 1;
        if (r < n1)
            return n1 + __left_->print(l, l) + (__right_ ? __right_->print(l, l) : 0);
        ptrdiff_t sz1 = __left_->print(f, l);
        if (r < n1 + sz1)
            return n1 + sz1 + (__right_ ? __right_->print(l, l) : 0);
        ptrdiff_t sz2 = __right_ ? __right_->print(f+sz1+1, l) : 0;
        if (r >= n1 + sz1 + sz2)
        {
            f += sz1;
            *f = '(';
            f += 1+sz2;
            *f = ')';
        }
        return n1 + sz1 + sz2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __delete_array_expr
    : public __node
{
public:

    __delete_array_expr(bool global, __node* op)
    {
        __size_ = global;
        __right_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = (__size_ ? 2 : 0) + 9 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__size_)
        {
            *buf++ = ':';
            *buf++ = ':';
        }
        strncpy(buf, "delete[] ", 9);
        return __right_->get_demangled_name(buf+9);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("delete[] ") - 1 + (__size_ ? 2 : 0);
        if (r < n1)
            return n1 + __right_->print(l, l);
        ptrdiff_t sz1 = __right_->print(f+n1, l);
        if (r >= n1 + sz1)
        {
            if (__size_)
            {
                *f++ = ':';
                *f++ = ':';
            }
            *f++ = 'd';
            *f++ = 'e';
            *f++ = 'l';
            *f++ = 'e';
            *f++ = 't';
            *f++ = 'e';
            *f++ = '[';
            *f++ = ']';
            *f   = ' ';
        }
        return n1 + sz1;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __delete_expr
    : public __node
{
public:

    __delete_expr(bool global, __node* op)
    {
        __size_ = global;
        __right_ = op;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = (__size_ ? 2 : 0) + 7 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__size_)
        {
            *buf++ = ':';
            *buf++ = ':';
        }
        strncpy(buf, "delete ", 7);
        return __right_->get_demangled_name(buf+7);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("delete ") - 1 + (__size_ ? 2 : 0);
        if (r < n1)
            return n1 + __right_->print(l, l);
        ptrdiff_t sz1 = __right_->print(f+n1, l);
        if (r >= n1 + sz1)
        {
            if (__size_)
            {
                *f++ = ':';
                *f++ = ':';
            }
            *f++ = 'd';
            *f++ = 'e';
            *f++ = 'l';
            *f++ = 'e';
            *f++ = 't';
            *f++ = 'e';
            *f   = ' ';
        }
        return n1 + sz1;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __new_expr
    : public __node
{
public:

    __new_expr(bool global, bool is_array, bool has_init,
               __node* expr, __node* type, __node* init)
    {
        __size_ =  (unsigned)global         |
                  ((unsigned)is_array << 1) |
                  ((unsigned)has_init << 2);
        __left_ = expr;
        __name_ = (const char*)type;
        __right_ = init;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            size_t off = 4;
            if (__size_ & 1)
                off += 2;
            if (__size_ & 2)
                off += 2;
            if (__left_)
            {
                off += 2;
                off += __left_->size();
            }
            __node* type = (__node*)__name_;
            off += type->size();
            if (__size_ & 4)
            {
                off += 2;
                if (__right_)
                    off += __right_->size();
            }
            const_cast<long&>(__cached_size_) = off;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__size_ & 1)
        {
            *buf++ = ':';
            *buf++ = ':';
        }
        *buf++ = 'n';
        *buf++ = 'e';
        *buf++ = 'w';
        if (__size_ & 2)
        {
            *buf++ = '[';
            *buf++ = ']';
        }
        if (__left_)
        {
            *buf++ = '(';
            buf = __left_->get_demangled_name(buf);
            *buf++ = ')';
        }
        *buf++ = ' ';
        __node* type = (__node*)__name_;
        buf = type->get_demangled_name(buf);
        if (__size_ & 4)
        {
            *buf++ = '(';
            if (__right_)
                buf = __right_->get_demangled_name(buf);
            *buf++ = ')';
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n1 = sizeof("new ") - 1 + (__size_ & 1 ? 2 : 0) +
              (__size_ & 2 ? 2 : 0) + (__left_ ? 2 : 0) + (__size_ & 4 ? 2 : 0);
        __node* type = (__node*)__name_;
        if (r < n1)
            return n1 + (__left_ ? __left_->print(l, l) : 0) +
                        type->print(l, l) +
                        (__right_ ? __right_->print(l, l) : 0);
        ptrdiff_t sz1 = __left_ ? __left_->print(f+4+
                                                 (__size_ & 1 ? 2 : 0) +
                                                 (__size_ & 2 ? 2 : 0), l) : 0;
        if (r < n1 + sz1)
            return n1 + sz1 + type->print(l, l) +
                              (__right_ ? __right_->print(l, l) : 0);
        ptrdiff_t sz2 = type->print(f+(n1-(__size_ & 4 ? 2 : 0)+sz1), l);
        if (r < n1 + sz1 + sz2)
            return n1 + sz1 + sz2 + (__right_ ? __right_->print(l, l) : 0);
        ptrdiff_t sz3 = __right_ ? __right_->print(f+(n1-1)+sz1+sz2, l) : 0;
        if (r >= n1 + sz1 + sz2 + sz3)
        {
            if (__size_ & 1)
            {
                *f++ = ':';
                *f++ = ':';
            }
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'w';
            if (__size_ & 2)
            {
                *f++ = '[';
                *f++ = ']';
            }
            if (__left_)
            {
                *f   = '(';
                f += 1 + sz1;
                *f++ = ')';
            }
            *f   = ' ';
            if (__size_ & 4)
            {
                f += 1 + sz2;
                *f = '(';
                f += 1 + sz3;
                *f = ')';
            }
        }
        return n1 + sz1 + sz2 + sz3;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        __node* type = (__node*)__name_;
        bool r = type->fix_forward_references(t_begin, t_end);
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);;
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);;
        return r;
    }
};

class __dot_star_expr
    : public __node
{
public:

    __dot_star_expr(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + 2 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        *buf++ = '.';
        *buf++ = '*';
        return __right_->get_demangled_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof(".*") - 1;
        if (r < n)
            return n + __left_->print(l, l) + __right_->print(l, l);
        ptrdiff_t sz1 = __left_->print(f, l);
        if (r < n + sz1)
            return n + sz1 + __right_->print(l, l);
        ptrdiff_t sz2 = __right_->print(f+sz1+n, l);
        if (r >= n + sz1 + sz2)
        {
            f += sz1;
            *f++ = '.';
            *f   = '*';
        }
        return n + sz1 + sz2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) &&
              __right_->fix_forward_references(t_begin, t_end);
    }
};

class __dot_expr
    : public __node
{
public:

    __dot_expr(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + 1 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        *buf++ = '.';
        return __right_->get_demangled_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof(".") - 1;
        if (r < n)
            return n + __left_->print(l, l) + __right_->print(l, l);
        ptrdiff_t sz1 = __left_->print(f, l);
        if (r < n + sz1)
            return n + sz1 + __right_->print(l, l);
        ptrdiff_t sz2 = __right_->print(f+sz1+n, l);
        if (r >= n + sz1 + sz2)
            f[sz1] = '.';
        return n + sz1 + sz2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) &&
              __right_->fix_forward_references(t_begin, t_end);
    }
};

class __arrow_expr
    : public __node
{
public:

    __arrow_expr(__node* op1, __node* op2)
    {
        __left_ = op1;
        __right_ = op2;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + 2 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        *buf++ = '-';
        *buf++ = '>';
        return __right_->get_demangled_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("->") - 1;
        if (r < n)
            return n + __left_->print(l, l) + __right_->print(l, l);
        ptrdiff_t sz1 = __left_->print(f, l);
        if (r < n + sz1)
            return n + sz1 + __right_->print(l, l);
        ptrdiff_t sz2 = __right_->print(f+sz1+n, l);
        if (r >= n + sz1 + sz2)
        {
            f += sz1;
            *f++ = '-';
            *f   = '>';
        }
        return n + sz1 + sz2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) &&
              __right_->fix_forward_references(t_begin, t_end);
    }
};

class __std_qualified_name
    : public __node
{
    static const ptrdiff_t n = sizeof("std") - 1;
public:

    __std_qualified_name()
    {
    }
    virtual size_t first_size() const
    {
        return n;
    }

    virtual char* first_demangled_name(char* buf) const
    {
        *buf++ = 's';
        *buf++ = 't';
        *buf++ = 'd';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 't';
            *f   = 'd';
        }
        return n;
    }
};

class __sub_allocator
    : public __node
{
    static const ptrdiff_t n = sizeof("std::allocator") - 1;
public:

    virtual size_t first_size() const
    {
        return n;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "std::allocator", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 't';
            *f++ = 'd';
            *f++ = ':';
            *f++ = ':';
            *f++ = 'a';
            *f++ = 'l';
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'c';
            *f++ = 'a';
            *f++ = 't';
            *f++ = 'o';
            *f   = 'r';
        }
        return n;
    }
};

class __sub_basic_string
    : public __node
{
    static const ptrdiff_t n = sizeof("std::basic_string") - 1;
public:

    virtual size_t first_size() const
    {
        return n;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "std::basic_string", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 't';
            *f++ = 'd';
            *f++ = ':';
            *f++ = ':';
            *f++ = 'b';
            *f++ = 'a';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'c';
            *f++ = '_';
            *f++ = 's';
            *f++ = 't';
            *f++ = 'r';
            *f++ = 'i';
            *f++ = 'n';
            *f   = 'g';
        }
        return n;
    }
};

class __sub_string
    : public __node
{
    static const size_t n = sizeof("std::string") - 1;
    static const size_t ne = sizeof("std::basic_string<char, std::char_traits<char>, std::allocator<char> >") - 1;
public:

    virtual size_t first_size() const
    {
        if (__size_)
            return ne;
        return n;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__size_)
        {
            strncpy(buf, "std::basic_string<char, std::char_traits<char>, std::allocator<char> >", ne);
            buf += ne;
        }
        else
        {
            strncpy(buf, "std::string", n);
            buf += n;
        }
        return buf;
    }

    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (__size_)
        {
            const ptrdiff_t n1 =
                sizeof("std::basic_string<char, std::char_traits<char>,"
                       " std::allocator<char> >") - 1;
            if (r >= n1)
                strncpy(f, "std::basic_string<char, std::char_traits<char>,"
                           " std::allocator<char> >", n1);
            return n1;
        }
        const ptrdiff_t n2 = sizeof("std::string") - 1;
        if (r >= n2)
        {
            *f++ = 's';
            *f++ = 't';
            *f++ = 'd';
            *f++ = ':';
            *f++ = ':';
            *f++ = 's';
            *f++ = 't';
            *f++ = 'r';
            *f++ = 'i';
            *f++ = 'n';
            *f   = 'g';
        }
        return n2;
    }
    virtual size_t base_size() const
    {
        return 12;
    }
    virtual char* get_base_name(char* buf) const
    {
        strncpy(buf, "basic_string", 12);
        return buf + 12;
    }
    virtual ptrdiff_t print_base_name(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("basic_string") - 1;
        if (r >= n)
        {
            *f++ = 'b';
            *f++ = 'a';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'c';
            *f++ = '_';
            *f++ = 's';
            *f++ = 't';
            *f++ = 'r';
            *f++ = 'i';
            *f++ = 'n';
            *f   = 'g';
        }
        return n;
    }

    virtual __node* base_name() const
    {
        const_cast<size_t&>(__size_) = true;
        return const_cast<__node*>(static_cast<const __node*>(this));
    }
};

class __sub_istream
    : public __node
{
    static const ptrdiff_t n = sizeof("std::istream") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "std::istream", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 't';
            *f++ = 'd';
            *f++ = ':';
            *f++ = ':';
            *f++ = 'i';
            *f++ = 's';
            *f++ = 't';
            *f++ = 'r';
            *f++ = 'e';
            *f++ = 'a';
            *f   = 'm';
        }
        return n;
    }
};

class __sub_ostream
    : public __node
{
    static const ptrdiff_t n = sizeof("std::ostream") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "std::ostream", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 't';
            *f++ = 'd';
            *f++ = ':';
            *f++ = ':';
            *f++ = 'o';
            *f++ = 's';
            *f++ = 't';
            *f++ = 'r';
            *f++ = 'e';
            *f++ = 'a';
            *f   = 'm';
        }
        return n;
    }
};

class __sub_iostream
    : public __node
{
    static const ptrdiff_t n = sizeof("std::iostream") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "std::iostream", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 't';
            *f++ = 'd';
            *f++ = ':';
            *f++ = ':';
            *f++ = 'i';
            *f++ = 'o';
            *f++ = 's';
            *f++ = 't';
            *f++ = 'r';
            *f++ = 'e';
            *f++ = 'a';
            *f   = 'm';
        }
        return n;
    }
};

class __sub
    : public __node
{
public:

    explicit __sub(__node* arg)
    {
        __left_ = arg;
    }
    explicit __sub(size_t arg)
    {
        __size_ = arg;
    }
    virtual size_t first_size() const
    {
        return __left_->first_size();
    }
    virtual char* first_demangled_name(char* buf) const
    {
        return __left_->first_demangled_name(buf);
    }
    virtual size_t second_size() const
    {
        return __left_->second_size();
    }
    virtual char* second_demangled_name(char* buf) const
    {
        return __left_->second_demangled_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        return __left_->print_first(f, l);
    }
    virtual ptrdiff_t print_second(char* f, char* l) const
    {
        return __left_->print_second(f, l);
    }
    virtual bool ends_with_template() const
    {
        return __left_->ends_with_template();
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool is_reference_or_pointer_to_function_or_array() const
    {
        return __left_->is_reference_or_pointer_to_function_or_array();
    }
    virtual bool is_function() const
    {
        return __left_->is_function();
    }
    virtual bool is_cv_qualifer() const
    {
        return __left_->is_cv_qualifer();
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return __left_->is_ctor_dtor_conv();
    }
    virtual bool is_array() const
    {
        return __left_->is_array();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        if (__left_ == 0)
        {
            if (__size_ < t_end - t_begin)
            {
                __left_ = t_begin[__size_];
                __size_ = 0;
            }
            else
                return false;
        }
        return true;
    }
    virtual size_t list_len() const
    {
        return __left_->list_len();
    }
    virtual bool is_sub() const
    {
        return true;
    }
};

class __unscoped_template_name
    : public __node
{
public:
    __unscoped_template_name(__node* name, __node* args)
        {__left_ = name; __right_ = args;}

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        return __right_->get_demangled_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        ptrdiff_t sz1 = __left_->print(f, l);
        if (r < sz1)
            return sz1 + __right_->print(l, l);
        return sz1 + __right_->print(f + sz1, l);
    }
    virtual bool ends_with_template() const
    {
        return __right_->ends_with_template();
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) && 
               __right_->fix_forward_references(t_begin, t_end);
    }
};

// length == 0: __left_ == NULL
// length == 1: __left_ != NULL, __right_ == NULL
// length  > 1: __left_ != NULL, __right_ != NULL
class __list
    : public __node
{
public:
    explicit __list(__node* type)
        {__left_ = type;}

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            if (__left_ == NULL)
                const_cast<long&>(__cached_size_) = 0;
            else if (__right_ == NULL)
                const_cast<long&>(__cached_size_) = __left_->size();
            else
            {
                size_t off = __right_->size();
                if (off > 0)
                    off += 2;
                const_cast<long&>(__cached_size_) = __left_->size() + off;
            }
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_ != NULL)
        {
            char* t = __left_->get_demangled_name(buf + (__size_ ? 2 : 0));
            if (__size_ == 0)
                buf = t;
            else if (t != buf+2)
            {
                *buf++ = ',';
                *buf++ = ' ';
                buf = t;
            }
            if (__right_)
                buf = __right_->get_demangled_name(buf);
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        if (__left_ == 0)
            return 0;
        const ptrdiff_t r = l - f;
        ptrdiff_t n = 0;
        if (__size_)
        {
            n = 2;
            if (r < n)
            {
                ptrdiff_t sz1 = __left_->print(l, l);
                if (sz1 == 0)
                    n = 0;
                return n + sz1 + (__right_ ? __right_->print(l, l) : 0);
            }
        }
        const ptrdiff_t sz1 = __left_->print(f+n, l);
        if (sz1 == 0)
            n = 0;
        else if (n != 0)
        {
            f[0] = ',';
            f[1] = ' ';
        }
        const ptrdiff_t sz2 = __right_ ? __right_->print(f+std::min(n+sz1, r), l) : 0;
        return n + sz1 + sz2;
    }
    virtual bool ends_with_template() const
    {
        if (__right_ != NULL)
            return __right_->ends_with_template();
        if (__left_ != NULL)
            return __left_->ends_with_template();
        return false;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
    virtual size_t list_len() const
    {
        if (!__left_)
            return 0;
        if (!__right_)
            return 1;
        return 1 + __right_->list_len();
    }
};

class __template_args
    : public __node
{
public:
    __template_args(__node* name, __node* list)
    {
        __left_ = name;
        __right_ = list;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            size_t off = 2;
            if (__right_)
            {
                if (__right_->ends_with_template())
                    ++off;
                off += __right_->size();
            }
            const_cast<long&>(__cached_size_) = __left_->size() + off;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        *buf++ = '<';
        if (__right_)
        {
            buf = __right_->get_demangled_name(buf);
            if (buf[-1] == '>')
                *buf++ = ' ';
        }
        *buf++ = '>';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz1 = __left_->print(f, l);
        ptrdiff_t sz2 = 0;
        ptrdiff_t n = 2;
        if (__right_)
        {
            sz2 = __right_->print(f+std::min(sz1+1, r), l);
            if (r >= sz1 + sz2 + 2)
            {
                if (f[sz1+sz2] == '>')
                {
                    f[sz1+sz2+1] = ' ';
                    ++n;
                }
            }
            else if (__right_->ends_with_template())
                ++n;
        }
        if (r >= sz1 + sz2 + n)
        {
            f[sz1] = '<';
            f[sz1+sz2+n-1] = '>';
        }
        return n + sz1 + sz2;
    }

    virtual bool ends_with_template() const
    {
        return true;
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return __left_->is_ctor_dtor_conv();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __function_args
    : public __node
{
public:
    __function_args(__node* list)
        {__right_ = list;}

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = 2 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        *buf++ = '(';
        buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = 2;
        if (r < n)
            return n + __right_->print(l, l);
        ptrdiff_t sz1 = __right_->print(f+1, l);
        if (r >= n + sz1)
        {
            *f = '(';
            f += 1 + sz1;
            *f = ')';
        }
        return n + sz1;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __cv_qualifiers
    : public __node
{
public:
    __cv_qualifiers(size_t cv, __node* type)
    {
        __left_ = type;
        __size_ = __left_->is_function() ? cv << 5 : cv;
    }

    virtual size_t first_size() const
    {
        size_t s = __left_->first_size();
        if (__size_ & 4)
            s += sizeof(" restrict")-1;
        if (__size_ & 2)
            s += sizeof(" volatile")-1;
        if (__size_ & 1)
            s += sizeof(" const")-1;
        if (__size_ & 8)
            s += sizeof(" &")-1;
        if (__size_ & 16)
            s += sizeof(" &&")-1;
        return s;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->first_demangled_name(buf);
        if (__size_ & 1)
        {
            const size_t n = sizeof(" const")-1;
            strncpy(buf, " const", n);
            buf += n;
        }
        if (__size_ & 2)
        {
            const size_t n = sizeof(" volatile")-1;
            strncpy(buf, " volatile", n);
            buf += n;
        }
        if (__size_ & 4)
        {
            const size_t n = sizeof(" restrict")-1;
            strncpy(buf, " restrict", n);
            buf += n;
        }
        if (__size_ & 8)
        {
            *buf++ = ' ';
            *buf++ = '&';
        }
        if (__size_ & 16)
        {
            *buf++ = ' ';
            *buf++ = '&';
            *buf++ = '&';
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz = __left_->print_first(f, l);
        ptrdiff_t n = 0;
        if (__size_ & 0x1F)
        {
            if (__size_ & 1)
            {
                const ptrdiff_t d = sizeof(" const")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t++ = 'c';
                    *t++ = 'o';
                    *t++ = 'n';
                    *t++ = 's';
                    *t   = 't';
                }
                n += d;
            }
            if (__size_ & 2)
            {
                const ptrdiff_t d = sizeof(" volatile")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t++ = 'v';
                    *t++ = 'o';
                    *t++ = 'l';
                    *t++ = 'a';
                    *t++ = 't';
                    *t++ = 'i';
                    *t++ = 'l';
                    *t   = 'e';
                }
                n += d;
            }
            if (__size_ & 4)
            {
                const ptrdiff_t d = sizeof(" restrict")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t++ = 'r';
                    *t++ = 'e';
                    *t++ = 's';
                    *t++ = 't';
                    *t++ = 'r';
                    *t++ = 'i';
                    *t++ = 'c';
                    *t   = 't';
                }
                n += d;
            }
            if (__size_ & 8)
            {
                const ptrdiff_t d = sizeof(" &")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t   = '&';
                }
                n += d;
            }
            if (__size_ & 16)
            {
                const ptrdiff_t d = sizeof(" &&")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t++ = '&';
                    *t   = '&';
                }
                n += d;
            }
        }
        return n + sz;
    }
    virtual size_t second_size() const
    {
        size_t s = __left_->second_size();
        if (__size_ & 128)
            s += sizeof(" restrict")-1;
        if (__size_ & 64)
            s += sizeof(" volatile")-1;
        if (__size_ & 32)
            s += sizeof(" const")-1;
        if (__size_ & 256)
            s += sizeof(" &")-1;
        if (__size_ & 512)
            s += sizeof(" &&")-1;
        return s;
    }
    virtual char* second_demangled_name(char* buf) const
    {
        buf = __left_->second_demangled_name(buf);
        if (__size_ & 32)
        {
            const size_t n = sizeof(" const")-1;
            strncpy(buf, " const", n);
            buf += n;
        }
        if (__size_ & 64)
        {
            const size_t n = sizeof(" volatile")-1;
            strncpy(buf, " volatile", n);
            buf += n;
        }
        if (__size_ & 128)
        {
            const size_t n = sizeof(" restrict")-1;
            strncpy(buf, " restrict", n);
            buf += n;
        }
        if (__size_ & 256)
        {
            *buf++ = ' ';
            *buf++ = '&';
        }
        if (__size_ & 512)
        {
            *buf++ = ' ';
            *buf++ = '&';
            *buf++ = '&';
        }
        return buf;
    }
    virtual ptrdiff_t print_second(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz = __left_->print_second(f, l);
        ptrdiff_t n = 0;
        if (__size_ & 0x3E0)
        {
            if (__size_ & 32)
            {
                const ptrdiff_t d = sizeof(" const")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t++ = 'c';
                    *t++ = 'o';
                    *t++ = 'n';
                    *t++ = 's';
                    *t   = 't';
                }
                n += d;
            }
            if (__size_ & 64)
            {
                const ptrdiff_t d = sizeof(" volatile")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t++ = 'v';
                    *t++ = 'o';
                    *t++ = 'l';
                    *t++ = 'a';
                    *t++ = 't';
                    *t++ = 'i';
                    *t++ = 'l';
                    *t   = 'e';
                }
                n += d;
            }
            if (__size_ & 128)
            {
                const ptrdiff_t d = sizeof(" restrict")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t++ = 'r';
                    *t++ = 'e';
                    *t++ = 's';
                    *t++ = 't';
                    *t++ = 'r';
                    *t++ = 'i';
                    *t++ = 'c';
                    *t   = 't';
                }
                n += d;
            }
            if (__size_ & 256)
            {
                const ptrdiff_t d = sizeof(" &")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t   = '&';
                }
                n += d;
            }
            if (__size_ & 512)
            {
                const ptrdiff_t d = sizeof(" &&")-1;
                if (r >= sz + n + d)
                {
                    char* t = f + sz + n;
                    *t++ = ' ';
                    *t++ = '&';
                    *t   = '&';
                }
                n += d;
            }
        }
        return n + sz;
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool is_reference_or_pointer_to_function_or_array() const
    {
        return __left_->is_reference_or_pointer_to_function_or_array();
    }
    virtual bool is_function() const
    {
        return __left_->is_function();
    }
    virtual bool is_cv_qualifer() const
    {
        return true;
    }
    virtual __node* extract_cv(__node*& rt) const
    {
        if (rt == this)
        {
            rt = __left_;
            return const_cast<__node*>(static_cast<const __node*>(this));
        }
        return 0;
    }
    virtual bool ends_with_template() const
    {
        return __left_->ends_with_template();
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return __left_->is_ctor_dtor_conv();
    }
    virtual bool is_array() const
    {
        return __left_->is_array();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
    virtual size_t list_len() const
    {
        return __left_->list_len();
    }
};

class __extended_qualifier
    : public __node
{
public:
    __extended_qualifier(__node* name, __node* type)
    {
        __left_ = type;
        __right_ = name;
        __size_ = __left_->is_function() ? 1 : 0;
    }

    virtual size_t first_size() const
    {
        size_t s = __left_->first_size();
        if (__size_ == 0)
            s += __right_->size() + 1;
        return s;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->first_demangled_name(buf);
        if (__size_ == 0)
        {
            *buf++ = ' ';
            buf = __right_->get_demangled_name(buf);
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz1 = __left_->print_first(f, l);
        ptrdiff_t sz2 = 0;
        ptrdiff_t n = 0;
        if (__size_ == 0)
        {
            if (r < sz1 + 1)
                return sz1 + 1 + __right_->print(l, l);
            sz2 = __right_->print(f+1+sz1, l);
            n = 1;
            f[sz1] = ' ';
        }
        return n + sz1 + sz2;
    }
    virtual size_t second_size() const
    {
        size_t s = __left_->second_size();
        if (__size_ == 1)
            s += __right_->size() + 1;
        return s;
    }
    virtual char* second_demangled_name(char* buf) const
    {
        buf = __left_->second_demangled_name(buf);
        if (__size_ == 1)
        {
            *buf++ = ' ';
            buf = __right_->get_demangled_name(buf);
        }
        return buf;
    }
    virtual ptrdiff_t print_second(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz1 = __left_->print_second(f, l);
        ptrdiff_t sz2 = 0;
        ptrdiff_t n = 0;
        if (__size_ == 1)
        {
            if (r < sz1 + 1)
                return sz1 + 1 + __right_->print(l, l);
            sz2 = __right_->print(f+1+sz1, l);
            n = 1;
            f[sz1] = ' ';
        }
        return n + sz1 + sz2;
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool is_reference_or_pointer_to_function_or_array() const
    {
        return __left_->is_reference_or_pointer_to_function_or_array();
    }
    virtual bool is_function() const
    {
        return __left_->is_function();
    }
    virtual bool is_cv_qualifer() const
    {
        return true;
    }
    virtual __node* extract_cv(__node*& rt) const
    {
        if (rt == this)
        {
            rt = __left_;
            return const_cast<__node*>(static_cast<const __node*>(this));
        }
        return 0;
    }
    virtual bool ends_with_template() const
    {
        return __left_->ends_with_template();
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return __left_->is_ctor_dtor_conv();
    }
    virtual bool is_array() const
    {
        return __left_->is_array();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
    virtual size_t list_len() const
    {
        return __left_->list_len();
    }
};

class __function
    : public __node
{
public:

    __function(__node* name, __node* signature, size_t ret_goes_first = true)
    {
        __size_ = ret_goes_first;
        __left_ = name;
        __right_ = signature;
    }

    virtual size_t first_size() const
    {
        size_t off = 0;
        if (__size_)
        {
            off = __right_->first_size();
            if (off > 0 && (__left_ == NULL ||
                            !__right_->__left_->is_reference_or_pointer_to_function_or_array()))
                ++off;
        }
        else
            off = 5;
        if (__left_)
            off += __left_->first_size();
        else
            ++off;
        return off;
    }

    virtual size_t second_size() const
    {
        size_t off = 0;
        if (__left_ == NULL)
            off = 1;
        off += __right_->second_size();
        if (!__size_)
        {
            off += 2;
            off += __right_->first_size();
        }
        return off;
    }

    virtual char* first_demangled_name(char* buf) const
    {
        if (__size_)
        {
            const char* t = buf;
            buf = __right_->first_demangled_name(buf);
            if (buf != t && (__left_ == NULL ||
                            !__right_->__left_->is_reference_or_pointer_to_function_or_array()))
                *buf++ = ' ';
        }
        else
        {
            strncpy(buf, "auto ", 5);
            buf += 5;
        }
        if (__left_)
            buf = __left_->first_demangled_name(buf);
        else
            *buf++ = '(';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        ptrdiff_t n = 0;
        ptrdiff_t sz1 = 0;
        ptrdiff_t sz2 = 0;
        if (__size_)
        {
            sz1 = __right_->print_first(f, l);
            if (sz1 != 0 && (__left_ == NULL ||
                            !__right_->__left_->is_reference_or_pointer_to_function_or_array()))
            {
                ++n;
                if (r >= sz1 + 1)
                    f[sz1] = ' ';
            }
        }
        else
        {
            n = 5;
            if (r >= 5)
            {
                char* t = f;
                *t++ = 'a';
                *t++ = 'u';
                *t++ = 't';
                *t++ = 'o';
                *t++ = ' ';
            }
        }
        if (__left_)
            sz2 = __left_->print_first(f + std::min(n + sz1, r), l);
        else
        {
            ++n;
            if (r >= n + sz1)
                f[n+sz1-1] = '(';
        }
        return n + sz1 + sz2;
    }

    virtual char* second_demangled_name(char* buf) const
    {
        if (__left_ == NULL)
            *buf++ = ')';
        buf = __right_->second_demangled_name(buf);
        if (!__size_)
        {
            *buf++ = '-';
            *buf++ = '>';
            buf = __right_->first_demangled_name(buf);
        }
        return buf;
    }
    virtual ptrdiff_t print_second(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        ptrdiff_t n = 0;
        ptrdiff_t sz1 = 0;
        ptrdiff_t sz2 = 0;
        if (__left_ == NULL)
        {
            n = 1;
            if (r >= 1)
                *f = ')';
        }
        sz1 = __right_->print_second(f+std::min(r, n), l);
        if (!__size_)
        {
            if (r > n+sz1+1)
            {
                f[n+sz1]   = '-';
                f[n+sz1+1] = '>';
            }
            n += 2;
            sz2 = __right_->print_first(f+std::min(r, n+sz1), l);
        }
        return n + sz1 + sz2;
    }

    virtual char* get_demangled_name(char* buf) const
    {
        if (__size_)
        {
            const char* t = buf;
            buf = __right_->first_demangled_name(buf);
            if (buf != t && (__left_ == NULL ||
                            !__right_->__left_->is_reference_or_pointer_to_function_or_array()))
                *buf++ = ' ';
        }
        else
        {
            strncpy(buf, "auto ", 5);
            buf += 5;
        }
        if (__left_)
            buf = __left_->first_demangled_name(buf);
        buf = __right_->second_demangled_name(buf);
        if (!__size_)
        {
            *buf++ = '-';
            *buf++ = '>';
            buf = __right_->first_demangled_name(buf);
        }
        return buf;
    }

    virtual size_t size() const
    {
        if (__cached_size_ == -1)
        {
            size_t off = 0;
            if (__size_)
            {
                off = __right_->first_size();
                if (off > 0 && (__left_ == NULL ||
                                !__right_->__left_->is_reference_or_pointer_to_function_or_array()))
                    ++off;
            }
            else
                off = 5;
            if (__left_)
                off += __left_->first_size();
            off += __right_->second_size();
            if (!__size_)
            {
                off += 2;
                off += __right_->first_size();
            }
            const_cast<long&>(__cached_size_) = off;
        }
        return __cached_size_;
    }

    virtual ptrdiff_t print(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        ptrdiff_t n = 0;
        ptrdiff_t sz1 = 0;
        ptrdiff_t sz2 = 0;
        if (__size_)
        {
            sz1 = __right_->print_first(f, l);
            if (sz1 != 0 && (__left_ == NULL ||
                            !__right_->__left_->is_reference_or_pointer_to_function_or_array()))
            {
                ++n;
                if (r >= sz1 + 1)
                    f[sz1] = ' ';
            }
        }
        else
        {
            n = 5;
            if (r >= 5)
            {
                char* t = f;
                *t++ = 'a';
                *t++ = 'u';
                *t++ = 't';
                *t++ = 'o';
                *t++ = ' ';
            }
        }
        if (__left_)
            sz2 = __left_->print_first(f + std::min(n + sz1, r), l);
        n += sz1 + sz2;
        sz2 = 0;
        sz1 = __right_->print_second(f+std::min(r, n), l);
        if (!__size_)
        {
            if (r > n+sz1+1)
            {
                f[n+sz1]   = '-';
                f[n+sz1+1] = '>';
            }
            n += 2;
            sz2 = __right_->print_first(f+std::min(r, n+sz1), l);
        }
        return n + sz1 + sz2;
    }

    virtual bool is_function() const
    {
        return true;
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return __left_->is_ctor_dtor_conv();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __function_signature
    : public __node
{
public:
    __function_signature(__node* ret, __node* args)
    {
        __left_ = ret;
        __right_ = args;
    }
    virtual size_t first_size() const
    {
        return __left_ ? __left_->first_size() : 0;
    }

    virtual size_t second_size() const
    {
        return 2 + (__right_ ? __right_->size() : 0)
                 + (__left_ ? __left_->second_size() : 0);
    }

    virtual char* first_demangled_name(char* buf) const
    {
        if (__left_)
            buf = __left_->first_demangled_name(buf);
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        return __left_ ? __left_->print_first(f, l) : 0;
    }

    virtual char* second_demangled_name(char* buf) const
    {
        *buf++ = '(';
        if (__right_)
            buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        if (__left_)
            buf = __left_->second_demangled_name(buf);
        return buf;
    }
    virtual ptrdiff_t print_second(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz1 = __right_ ? __right_->print(f+std::min<ptrdiff_t>(1, r), l) : 0;
        const ptrdiff_t sz2 = __left_ ? __left_->print_second(f+std::min(2+sz1, r), l) : 0;
        if (r >= 2 + sz1 + sz2)
        {
            *f = '(';
            f += 1 + sz1;
            *f = ')';
        }
        return 2 + sz1 + sz2;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = r && __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __pointer_to
    : public __node
{
public:

    explicit __pointer_to(__node* type)
    {
        __left_ = type;
    }
    virtual size_t first_size() const
    {
        return __left_->first_size() + (__left_->is_array() ? 3 : 1);
    }
    virtual size_t second_size() const
    {
        return __left_->second_size() + (__left_->is_array() ? 1 : 0);
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->first_demangled_name(buf);
        if (__left_->is_array())
        {
            *buf++ = ' ';
            *buf++ = '(';
            *buf++ = '*';
        }
        else
            *buf++ = '*';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz = __left_->print_first(f, l);
        ptrdiff_t n;
        if (__left_->is_array())
        {
            n = 3;
            if (r >= sz + n)
            {
                f += sz;
                *f++ = ' ';
                *f++ = '(';
                *f   = '*';
            }
        }
        else
        {
            n = 1;
            if (r >= sz + n)
                f[sz] = '*';
        }
        return sz + n;
    }
    virtual char* second_demangled_name(char* buf) const
    {
        if (__left_->is_array())
            *buf++ = ')';
        return __left_->second_demangled_name(buf);
    }
    virtual ptrdiff_t print_second(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        ptrdiff_t n = 0;
        if (__left_->is_array())
        {
            n = 1;
            if (r > n)
                *f = ')';
        }
        return __left_->print_second(f + std::min(n, r), l) + n;
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool is_reference_or_pointer_to_function_or_array() const
    {
        return __left_->is_function() ||
               __left_->is_reference_or_pointer_to_function_or_array();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
    virtual size_t list_len() const
    {
        return __left_->list_len();
    }
};

class __lvalue_reference_to
    : public __node
{
public:

    explicit __lvalue_reference_to(__node* type)
    {
        __left_ = type;
    }
    virtual size_t first_size() const
    {
        return __left_->first_size() + (__left_->is_array() ? 3 : 1);
    }
    virtual size_t second_size() const
    {
        return __left_->second_size() + (__left_->is_array() ? 1 : 0);
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->first_demangled_name(buf);
        if (__left_->is_array())
        {
            *buf++ = ' ';
            *buf++ = '(';
            *buf++ = '&';
        }
        else
            *buf++ = '&';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz = __left_->print_first(f, l);
        ptrdiff_t n;
        if (__left_->is_array())
        {
            n = 3;
            if (r >= sz + n)
            {
                f += sz;
                *f++ = ' ';
                *f++ = '(';
                *f   = '&';
            }
        }
        else
        {
            n = 1;
            if (r >= sz + n)
                f[sz] = '&';
        }
        return sz + n;
    }
    virtual char* second_demangled_name(char* buf) const
    {
        if (__left_->is_array())
            *buf++ = ')';
        return __left_->second_demangled_name(buf);
    }
    virtual ptrdiff_t print_second(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        ptrdiff_t n = 0;
        if (__left_->is_array())
        {
            n = 1;
            if (r > n)
                *f = ')';
        }
        return __left_->print_second(f + std::min(n, r), l) + n;
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool is_reference_or_pointer_to_function_or_array() const
    {
        return __left_->is_function();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
    virtual size_t list_len() const
    {
        return __left_->list_len();
    }
};

class __rvalue_reference_to
    : public __node
{
public:

    explicit __rvalue_reference_to(__node* type)
    {
        __left_ = type;
    }
    virtual size_t first_size() const
    {
        return __left_->first_size() + (__left_->is_array() ? 4 : 2);
    }
    virtual size_t second_size() const
    {
        return __left_->second_size() + (__left_->is_array() ? 1 : 0);
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->first_demangled_name(buf);
        if (__left_->is_array())
        {
            strncpy(buf, " (&&", 4);
            buf += 4;
        }
        else
        {
            *buf++ = '&';
            *buf++ = '&';
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz = __left_->print_first(f, l);
        ptrdiff_t n;
        if (__left_->is_array())
        {
            n = 4;
            if (r >= sz + n)
            {
                f += sz;
                *f++ = ' ';
                *f++ = '(';
                *f++ = '&';
                *f   = '&';
            }
        }
        else
        {
            n = 2;
            if (r >= sz + n)
            {
                f += sz;
                *f++ = '&';
                *f   = '&';
            }
        }
        return sz + n;
    }
    virtual char* second_demangled_name(char* buf) const
    {
        if (__left_->is_array())
            *buf++ = ')';
        return __left_->second_demangled_name(buf);
    }
    virtual ptrdiff_t print_second(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        ptrdiff_t n = 0;
        if (__left_->is_array())
        {
            n = 1;
            if (r > n)
                *f = ')';
        }
        return __left_->print_second(f + std::min(n, r), l) + n;
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool is_reference_or_pointer_to_function_or_array() const
    {
        return __left_->is_function();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
    virtual size_t list_len() const
    {
        return __left_->list_len();
    }
};

class __d_complex
    : public __node
{
    static const size_t n = sizeof(" complex") - 1;
public:

    explicit __d_complex(__node* type)
    {
        __left_ = type;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __left_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        strncpy(buf, " complex", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz = __left_->print(f, l);
        const ptrdiff_t n = sizeof(" complex") - 1;
        if (r >= sz + n)
        {
            f += sz;
            *f++ = ' ';
            *f++ = 'c';
            *f++ = 'o';
            *f++ = 'm';
            *f++ = 'p';
            *f++ = 'l';
            *f++ = 'e';
            *f   = 'x';
        }
        return sz + n;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
};

class __imaginary
    : public __node
{
    static const size_t n = sizeof(" imaginary") - 1;
public:

    explicit __imaginary(__node* type)
    {
        __left_ = type;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = n + __left_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        strncpy(buf, " imaginary", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t sz = __left_->print(f, l);
        const ptrdiff_t n = sizeof(" imaginary") - 1;
        if (r >= sz + n)
        {
            f += sz;
            *f++ = ' ';
            *f++ = 'i';
            *f++ = 'm';
            *f++ = 'a';
            *f++ = 'g';
            *f++ = 'i';
            *f++ = 'n';
            *f++ = 'a';
            *f++ = 'r';
            *f   = 'y';
        }
        return sz + n;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
};

class __pack_expansion
    : public __node
{
public:

    explicit __pack_expansion(__node* type)
    {
        __left_ = type;
    }
    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            size_t len = __left_->list_len();
            size_t off = 0;
            if (len != 0)
            {
                if (__left_->is_sub() || len == 1)
                    off = __left_->size();
                else
                {
                    __node* top = __left_;
                    __node* bottom = top;
                    while (!bottom->__left_->is_sub())
                        bottom = bottom->__left_;
                    __node* sub = bottom->__left_;
                    __node* i = sub->__left_;
                    bool first = true;
                    top->reset_cached_size();
                    while (i)
                    {
                        if (!first)
                            off += 2;
                        bottom->__left_ = i->__left_;
                        off += top->size();
                        top->reset_cached_size();
                        i = i->__right_;
                        first = false;
                    }
                    bottom->__left_ = sub;
                }
            }
            const_cast<long&>(__cached_size_) = off;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        size_t len = __left_->list_len();
        if (len != 0)
        {
            if (__left_->is_sub() || len == 1)
                buf = __left_->get_demangled_name(buf);
            else
            {
                __node* top = __left_;
                __node* bottom = top;
                while (!bottom->__left_->is_sub())
                    bottom = bottom->__left_;
                __node* sub = bottom->__left_;
                __node* i = sub->__left_;
                bool first = true;
                top->reset_cached_size();
                while (i)
                {
                    if (!first)
                    {
                        *buf++ = ',';
                        *buf++ = ' ';
                    }
                    bottom->__left_ = i->__left_;
                    buf = top->get_demangled_name(buf);
                    top->reset_cached_size();
                    i = i->__right_;
                    first = false;
                }
                bottom->__left_ = sub;
            }
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t len = __left_->list_len();
        ptrdiff_t sz = 0;
        if (len != 0)
        {
             if (__left_->is_sub() || len == 1)
                sz = __left_->print(f, l);
            else
            {
                __node* top = __left_;
                __node* bottom = top;
                while (!bottom->__left_->is_sub())
                    bottom = bottom->__left_;
                __node* sub = bottom->__left_;
                __node* i = sub->__left_;
                bool first = true;
                while (i)
                {
                    if (!first)
                    {
                        if (r >= sz+2)
                        {
                            f[sz]   = ',';
                            f[sz+1] = ' ';
                        }
                        sz += 2;
                    }
                    bottom->__left_ = i->__left_;
                    sz += top->print(f+std::min(sz, r), l);
                    i = i->__right_;
                    first = false;
                }
                bottom->__left_ = sub;
            }
        }
        return sz;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
};

class __void
    : public __node
{
    static const size_t n = sizeof("void") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "void", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'v';
            *f++ = 'o';
            *f++ = 'i';
            *f   = 'd';
        }
        return n;
    }
};

class __wchar_t
    : public __node
{
    static const size_t n = sizeof("wchar_t") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "wchar_t", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'w';
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = '_';
            *f   = 't';
        }
        return n;
    }
};

class __wchar_t_literal
    : public __node
{
public:
    explicit __wchar_t_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+9;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "(wchar_t)", 9);
        buf += 9;
        strncpy(buf, __name_, __size_);
        return buf + __size_;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("(wchar_t)") - 1;
        if (r >= n + __size_)
        {
            *f++ = '(';
            *f++ = 'w';
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = '_';
            *f++ = 't';
            *f++ = ')';
            strncpy(f, __name_, __size_);
        }
        return n + __size_;
    }
};

class __bool
    : public __node
{
    static const size_t n = sizeof("bool") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "bool", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'b';
            *f++ = 'o';
            *f++ = 'o';
            *f   = 'l';
        }
        return n;
    }
};

class __bool_literal
    : public __node
{
public:
    explicit __bool_literal(const char* __name, unsigned __size)
    {
        __name_ = __name;
        __size_ = __size;
    }

    virtual size_t first_size() const
    {
        return __size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, __name_, __size_);
        return buf + __size_;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= __size_)
            strncpy(f, __name_, __size_);
        return __size_;
    }
};

class __char
    : public __node
{
    static const size_t n = sizeof("char") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "char", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f   = 'r';
        }
        return n;
    }
};

class __char_literal
    : public __node
{
public:
    explicit __char_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+6;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "(char)", 6);
        buf += 6;
        if (*__name_ == 'n')
        {
            *buf++ = '-';  // strncpy(buf+6, "-", 1);
            strncpy(buf, __name_+1, __size_-1);
            buf += __size_ - 1;
        }
        else
        {
            strncpy(buf, __name_, __size_);
            buf += __size_;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("(char)") - 1;
        if (r >= __size_ + n)
        {
            *f++ = '(';
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = ')';
            if (*__name_ == 'n')
            {
                *f++ = '-';
                strncpy(f, __name_+1, __size_-1);
            }
            else
                strncpy(f, __name_, __size_);
        }
        return __size_ + n;
    }
};

class __signed_char
    : public __node
{
    static const size_t n = sizeof("signed char") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "signed char", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f   = 'r';
        }
        return n;
    }
};

class __signed_char_literal
    : public __node
{
public:
    explicit __signed_char_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+13;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "(signed char)", 13);
        buf += 13;
        if (*__name_ == 'n')
        {
            *buf++ = '-';
            strncpy(buf, __name_+1, __size_-1);
            buf += __size_ - 1;
        }
        else
        {
            strncpy(buf, __name_, __size_);
            buf += __size_;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("(signed char)") - 1;
        if (r >= __size_ + n)
        {
            *f++ = '(';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = ')';
            if (*__name_ == 'n')
            {
                *f++ = '-';
                strncpy(f, __name_+1, __size_-1);
            }
            else
                strncpy(f, __name_, __size_);
        }
        return __size_ + n;
    }
};

class __unsigned_char
    : public __node
{
    static const size_t n = sizeof("unsigned char") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "unsigned char", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f   = 'r';
        }
        return n;
    }
};

class __unsigned_char_literal
    : public __node
{
public:
    explicit __unsigned_char_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+15;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "(unsigned char)", 15);
        buf += 15;
        strncpy(buf, __name_, __size_);
        return buf + __size_;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("(unsigned char)") - 1;
        if (r >= __size_ + n)
        {
            *f++ = '(';
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = ')';
            strncpy(f, __name_, __size_);
        }
        return __size_ + n;
    }
};

class __short
    : public __node
{
    static const size_t n = sizeof("short") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "short", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 'h';
            *f++ = 'o';
            *f++ = 'r';
            *f   = 't';
        }
        return n;
    }
};

class __short_literal
    : public __node
{
public:
    explicit __short_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+7;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "(short)", 7);
        buf += 7;
        if (*__name_ == 'n')
        {
            *buf++ = '-';
            strncpy(buf, __name_+1, __size_-1);
            buf += __size_ - 1;
        }
        else
        {
            strncpy(buf, __name_, __size_);
            buf += __size_;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("(short)") - 1;
        if (r >= __size_ + n)
        {
            *f++ = '(';
            *f++ = 's';
            *f++ = 'h';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = 't';
            *f++ = ')';
            if (*__name_ == 'n')
            {
                *f++ = '-';
                strncpy(f, __name_+1, __size_-1);
            }
            else
                strncpy(f, __name_, __size_);
        }
        return __size_ + n;
    }
};

class __unsigned_short
    : public __node
{
    static const size_t n = sizeof("unsigned short") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "unsigned short", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 's';
            *f++ = 'h';
            *f++ = 'o';
            *f++ = 'r';
            *f   = 't';
        }
        return n;
    }
};

class __unsigned_short_literal
    : public __node
{
public:
    explicit __unsigned_short_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+16;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "(unsigned short)", 16);
        buf += 16;
        strncpy(buf, __name_, __size_);
        return buf + __size_;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("(unsigned short)") - 1;
        if (r >= __size_ + n)
        {
            *f++ = '(';
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 's';
            *f++ = 'h';
            *f++ = 'o';
            *f++ = 'r';
            *f++ = 't';
            *f++ = ')';
            strncpy(f, __name_, __size_);
        }
        return __size_ + n;
    }
};

class __int
    : public __node
{
    static const size_t n = sizeof("int") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        *buf++ = 'i';
        *buf++ = 'n';
        *buf++ = 't';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'i';
            *f++ = 'n';
            *f   = 't';
        }
        return n;
    }
};

class __int_literal
    : public __node
{
public:
    explicit __int_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (*__name_ == 'n')
        {
            *buf++ = '-';
            strncpy(buf, __name_+1, __size_-1);
            buf += __size_ - 1;
        }
        else
        {
            strncpy(buf, __name_, __size_);
            buf += __size_;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= __size_)
        {
            if (*__name_ == 'n')
            {
                *f++ = '-';
                strncpy(f, __name_+1, __size_-1);
            }
            else
                strncpy(f, __name_, __size_);
        }
        return __size_;
    }
};

class __unsigned_int
    : public __node
{
    static const size_t n = sizeof("unsigned int") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "unsigned int", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 'i';
            *f++ = 'n';
            *f   = 't';
        }
        return n;
    }
};

class __unsigned_int_literal
    : public __node
{
public:
    explicit __unsigned_int_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+1;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, __name_, __size_);
        buf += __size_;
        *buf++ = 'u';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("u") - 1;
        if (r >= __size_ + n)
        {
            strncpy(f, __name_, __size_);
            f[__size_] = 'u';
        }
        return __size_ + n;
    }
};

class __long
    : public __node
{
    static const size_t n = sizeof("long") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "long", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'n';
            *f   = 'g';
        }
        return n;
    }
};

class __long_literal
    : public __node
{
public:
    explicit __long_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+1;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (*__name_ == 'n')
        {
            *buf++ = '-';  // strncpy(buf, "-", 1);
            strncpy(buf, __name_+1, __size_-1);
            buf += __size_ - 1;
        }
        else
        {
            strncpy(buf, __name_, __size_);
            buf += __size_;
        }
        *buf++ = 'l';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("l") - 1;
        if (r >= __size_ + n)
        {
            if (*__name_ == 'n')
            {
                *f++ = '-';
                strncpy(f, __name_+1, __size_-1);
                f += __size_-1;
            }
            else
            {
                strncpy(f, __name_, __size_);
                f += __size_;
            }
            *f = 'l';
        }
        return __size_ + n;
    }
};

class __unsigned_long
    : public __node
{
    static const size_t n = sizeof("unsigned long") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "unsigned long", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'n';
            *f   = 'g';
        }
        return n;
    }
};

class __unsigned_long_literal
    : public __node
{
public:
    explicit __unsigned_long_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+2;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, __name_, __size_);
        buf += __size_;
        *buf++ = 'u';
        *buf++ = 'l';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("ul") - 1;
        if (r >= __size_ + n)
        {
            strncpy(f, __name_, __size_);
            f += __size_;
            *f++ = 'u';
            *f   = 'l';
        }
        return __size_ + n;
    }
};

class __long_long
    : public __node
{
    static const size_t n = sizeof("long long") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "long long", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'n';
            *f++ = 'g';
            *f++ = ' ';
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'n';
            *f   = 'g';
        }
        return n;
    }
};

class __long_long_literal
    : public __node
{
public:
    explicit __long_long_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+2;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        if (*__name_ == 'n')
        {
            *buf++ = '-';
            strncpy(buf, __name_+1, __size_-1);
            buf += __size_ - 1;
        }
        else
        {
            strncpy(buf, __name_, __size_);
            buf += __size_;
        }
        *buf++ = 'l';
        *buf++ = 'l';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("ll") - 1;
        if (r >= __size_ + n)
        {
            if (*__name_ == 'n')
            {
                *f++ = '-';
                strncpy(f, __name_+1, __size_-1);
                f += __size_-1;
            }
            else
            {
                strncpy(f, __name_, __size_);
                f += __size_;
            }
            *f++ = 'l';
            *f   = 'l';
        }
        return __size_ + n;
    }
};

class __unsigned_long_long
    : public __node
{
    static const size_t n = sizeof("unsigned long long") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "unsigned long long", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'n';
            *f++ = 'g';
            *f++ = ' ';
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'n';
            *f   = 'g';
        }
        return n;
    }
};

class __unsigned_long_long_literal
    : public __node
{
public:
    explicit __unsigned_long_long_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+3;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, __name_, __size_);
        buf += __size_;
        *buf++ = 'u';
        *buf++ = 'l';
        *buf++ = 'l';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("ull") - 1;
        if (r >= __size_ + n)
        {
            strncpy(f, __name_, __size_);
            f += __size_;
            *f++ = 'u';
            *f++ = 'l';
            *f   = 'l';
        }
        return __size_ + n;
    }
};

class __int128
    : public __node
{
    static const size_t n = sizeof("__int128") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "__int128", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = '_';
            *f++ = '_';
            *f++ = 'i';
            *f++ = 'n';
            *f++ = 't';
            *f++ = '1';
            *f++ = '2';
            *f   = '8';
        }
        return n;
    }
};

class __int128_literal
    : public __node
{
public:
    explicit __int128_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+10;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "(__int128)", 10);
        buf += 10;
        if (*__name_ == 'n')
        {
            *buf++ = '-';
            strncpy(buf, __name_+1, __size_-1);
            buf += __size_ - 1;
        }
        else
        {
            strncpy(buf, __name_, __size_);
            buf += __size_;
        }
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("(__int128)") - 1;
        if (r >= __size_ + n)
        {
            *f++ = '(';
            *f++ = '_';
            *f++ = '_';
            *f++ = 'i';
            *f++ = 'n';
            *f++ = 't';
            *f++ = '1';
            *f++ = '2';
            *f++ = '8';
            *f   = ')';
            if (*__name_ == 'n')
            {
                *f++ = '-';
                strncpy(f, __name_+1, __size_-1);
            }
            else
                strncpy(f, __name_, __size_);
        }
        return __size_ + n;
    }
};

class __unsigned_int128
    : public __node
{
    static const size_t n = sizeof("unsigned __int128") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "unsigned __int128", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = '_';
            *f++ = '_';
            *f++ = 'i';
            *f++ = 'n';
            *f++ = 't';
            *f++ = '1';
            *f++ = '2';
            *f   = '8';
        }
        return n;
    }
};

class __unsigned_int128_literal
    : public __node
{
public:
    explicit __unsigned_int128_literal(const char* __first, const char* __last)
    {
        __name_ = __first;
        __size_ = __last - __first;
    }

    virtual size_t first_size() const
    {
        return __size_+19;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "(unsigned __int128)", 19);
        buf += 19;
        strncpy(buf, __name_, __size_);
        return buf + __size_;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("(unsigned __int128)") - 1;
        if (r >= __size_ + n)
        {
            *f++ = '(';
            *f++ = 'u';
            *f++ = 'n';
            *f++ = 's';
            *f++ = 'i';
            *f++ = 'g';
            *f++ = 'n';
            *f++ = 'e';
            *f++ = 'd';
            *f++ = ' ';
            *f++ = '_';
            *f++ = '_';
            *f++ = 'i';
            *f++ = 'n';
            *f++ = 't';
            *f++ = '1';
            *f++ = '2';
            *f++ = '8';
            *f   = ')';
            strncpy(f, __name_, __size_);
        }
        return __size_ + n;
    }
};

class __float_literal
    : public __node
{
public:
    explicit __float_literal(float value)
    {
        __value_ = value;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            char num[20] = {0};
            float v = static_cast<float>(__value_);
            const_cast<long&>(__cached_size_) = sprintf(num, "%a", v)+1;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        char num[20] = {0};
        float v = static_cast<float>(__value_);
        int n = sprintf(num, "%a", v);
        strncpy(buf, num, n);
        buf += n;
        *buf++ = 'f';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        char num[20] = {0};
        float v = static_cast<float>(__value_);
        ptrdiff_t n = sprintf(num, "%a", v);
        if (r >= n+1)
        {
            strncpy(f, num, n);
            f[n] = 'f';
        }
        ++n;
        return n;
    }
};

class __float
    : public __node
{
    static const size_t n = sizeof("float") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "float", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'f';
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'a';
            *f   = 't';
        }
        return n;
    }
};

class __double_literal
    : public __node
{
public:
    explicit __double_literal(double value)
    {
        __value_ = value;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            char num[30] = {0};
            double v = static_cast<double>(__value_);
            const_cast<long&>(__cached_size_) = sprintf(num, "%a", v);
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        char num[30] = {0};
        double v = static_cast<double>(__value_);
        int n = sprintf(num, "%a", v);
        strncpy(buf, num, n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        char num[30] = {0};
        double v = static_cast<double>(__value_);
        const ptrdiff_t n = sprintf(num, "%a", v);
        if (r >= n)
            strncpy(f, num, n);
        return n;
    }
};

class __double
    : public __node
{
    static const size_t n = sizeof("double") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "double", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'd';
            *f++ = 'o';
            *f++ = 'u';
            *f++ = 'b';
            *f++ = 'l';
            *f   = 'e';
        }
        return n;
    }
};

class __long_double
    : public __node
{
    static const size_t n = sizeof("long double") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "long double", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'n';
            *f++ = 'g';
            *f++ = ' ';
            *f++ = 'd';
            *f++ = 'o';
            *f++ = 'u';
            *f++ = 'b';
            *f++ = 'l';
            *f   = 'e';
        }
        return n;
    }
};

class __float128
    : public __node
{
    static const size_t n = sizeof("__float128") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "__float128", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = '_';
            *f++ = '_';
            *f++ = 'f';
            *f++ = 'l';
            *f++ = 'o';
            *f++ = 'a';
            *f++ = 't';
            *f++ = '1';
            *f++ = '2';
            *f   = '8';
        }
        return n;
    }
};

class __ellipsis
    : public __node
{
    static const size_t n = sizeof("...") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        *buf++ = '.';
        *buf++ = '.';
        *buf++ = '.';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = '.';
            *f++ = '.';
            *f   = '.';
        }
        return n;
    }
};

class __decimal64
    : public __node
{
    static const size_t n = sizeof("decimal64") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "decimal64", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'd';
            *f++ = 'e';
            *f++ = 'c';
            *f++ = 'i';
            *f++ = 'm';
            *f++ = 'a';
            *f++ = 'l';
            *f++ = '6';
            *f   = '4';
        }
        return n;
    }
};

class __decimal128
    : public __node
{
    static const size_t n = sizeof("decimal128") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "decimal128", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'd';
            *f++ = 'e';
            *f++ = 'c';
            *f++ = 'i';
            *f++ = 'm';
            *f++ = 'a';
            *f++ = 'l';
            *f++ = '1';
            *f++ = '2';
            *f   = '8';
        }
        return n;
    }
};

class __decimal32
    : public __node
{
    static const size_t n = sizeof("decimal32") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "decimal32", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'd';
            *f++ = 'e';
            *f++ = 'c';
            *f++ = 'i';
            *f++ = 'm';
            *f++ = 'a';
            *f++ = 'l';
            *f++ = '3';
            *f   = '2';
        }
        return n;
    }
};

class __decimal16
    : public __node
{
    static const size_t n = sizeof("decimal16") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "decimal16", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'd';
            *f++ = 'e';
            *f++ = 'c';
            *f++ = 'i';
            *f++ = 'm';
            *f++ = 'a';
            *f++ = 'l';
            *f++ = '1';
            *f   = '6';
        }
        return n;
    }
};

class __d_char32_t
    : public __node
{
    static const size_t n = sizeof("char32_t") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "char32_t", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = '3';
            *f++ = '2';
            *f++ = '_';
            *f   = 't';
        }
        return n;
    }
};

class __d_char16_t
    : public __node
{
    static const size_t n = sizeof("char16_t") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "char16_t", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'c';
            *f++ = 'h';
            *f++ = 'a';
            *f++ = 'r';
            *f++ = '1';
            *f++ = '6';
            *f++ = '_';
            *f   = 't';
        }
        return n;
    }
};

class __auto
    : public __node
{
    static const size_t n = sizeof("auto") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "auto", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 'a';
            *f++ = 'u';
            *f++ = 't';
            *f   = 'o';
        }
        return n;
    }
};

class __nullptr_t
    : public __node
{
    static const size_t n = sizeof("std::nullptr_t") - 1;
public:

    virtual size_t first_size() const {return n;}
    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "std::nullptr_t", n);
        return buf + n;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 't';
            *f++ = 'd';
            *f++ = ':';
            *f++ = ':';
            *f++ = 'n';
            *f++ = 'u';
            *f++ = 'l';
            *f++ = 'l';
            *f++ = 'p';
            *f++ = 't';
            *f++ = 'r';
            *f++ = '_';
            *f   = 't';
        }
        return n;
    }
};

class __array
    : public __node
{
public:

    explicit __array(__node* type)
    {
        __left_ = type;
    }

    __array(__node* type, size_t dim)
    {
        __left_ = type;
        __size_ = dim;
    }

    __array(__node* type, __node* dim)
    {
        __left_ = type;
        __right_ = dim;
    }

    virtual size_t size() const
    {
        if (__cached_size_ == -1)
        {
            size_t r = __left_->size() + 3;
            if (__right_ != 0)
                r += __right_->size();
            else if (__size_ != 0)
                r += snprintf(0, 0, "%ld", __size_);
            const_cast<long&>(__cached_size_) = r;
        }
        return __cached_size_;
    }

    virtual char* get_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        *buf++ = ' ';
        *buf++ = '[';
        if (__right_ != 0)
            buf = __right_->get_demangled_name(buf);
        else if (__size_ != 0)
        {
            size_t rs = sprintf(buf, "%ld", __size_);
            buf += rs;
        }
        *buf++ = ']';
        return buf;
    }
    virtual ptrdiff_t print(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = 3;
        const ptrdiff_t sz1 = __left_->print(f, l);
        char buf[20];
        ptrdiff_t sz2 = 0;
        if (__right_ != 0)
            sz2 = __right_->print(f+std::min(sz1+(n-1), r), l);
        else if (__size_ != 0)
        {
            sz2 = sprintf(buf, "%ld", __size_);
            if (r >= sz1 + sz2 + n)
                strncpy(f+sz1+2, buf, sz2);
        }
        if (r >= sz1 + sz2 + n)
        {
            f += sz1;
            *f++ = ' ';
            *f   = '[';
            f += 1 + sz2;
            *f   = ']';
        }
        return sz1 + sz2 + n;
    }

    virtual size_t first_size() const
    {
        return __left_->first_size();
    }

    virtual char* first_demangled_name(char* buf) const
    {
        return __left_->first_demangled_name(buf);
    }

    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        return __left_->print_first(f, l);
    }

    virtual size_t second_size() const
    {
        size_t r = 2 + __left_->second_size();
        if (!__left_->is_array())
            ++r;
        if (__right_ != 0)
            r += __right_->size();
        else if (__size_ != 0)
            r += snprintf(0, 0, "%ld", __size_);
        return r;
    }

    virtual char* second_demangled_name(char* buf) const
    {
        *buf++ = ' ';
        *buf++ = '[';
        if (__right_ != 0)
            buf = __right_->get_demangled_name(buf);
        else if (__size_ != 0)
        {
            size_t off = sprintf(buf, "%ld", __size_);
            buf += off;
        }
        char* t = buf;
        buf = __left_->second_demangled_name(buf);
        *t = ']';
        if (buf == t)
            ++buf;
        return buf;
    }
    virtual ptrdiff_t print_second(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        ptrdiff_t n = 2;
        char buf[20];
        ptrdiff_t sz2 = 0;
        if (__right_ != 0)
            sz2 = __right_->print(f+std::min(n, r), l);
        else if (__size_ != 0)
        {
            sz2 = sprintf(buf, "%ld", __size_);
            if (r >= sz2 + 3)
                strncpy(f+2, buf, sz2);
        }
        const ptrdiff_t sz1 = __left_->print_second(f+std::min(2+sz2, r), l);
        if (sz1 == 0)
            ++n;
        if (r >= sz1 + sz2 + n)
        {
            *f++ = ' ';
            *f   = '[';
            f += 1 + sz2;
            *f   = ']';
        }
        return sz1 + sz2 + n;
    }
    virtual bool is_array() const
    {
        return true;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = __left_->fix_forward_references(t_begin, t_end);
        if (__right_)
            r = r && __right_->fix_forward_references(t_begin, t_end);
        return r;
    }
};

class __pointer_to_member_type
    : public __node
{
public:

    __pointer_to_member_type(__node* class_type, __node* member_type)
    {
        __left_ = class_type;
        __right_ = member_type;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + 3 + __right_->size();
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __right_->first_demangled_name(buf);
        buf = __left_->get_demangled_name(buf);
        *buf++ = ':';
        *buf++ = ':';
        *buf++ = '*';
        return __right_->second_demangled_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = 3;
        const ptrdiff_t sz1 = __right_->print_first(f, l);
        const ptrdiff_t sz2 = __left_->print(f+std::min(sz1, r), l);
        const ptrdiff_t sz3 = __right_->print_second(f+std::min(sz1+sz2+n, r), l);
        if (r >= sz1 + sz2 + sz3 + n)
        {
            f += sz1 + sz2;
            *f++ = ':';
            *f++ = ':';
            *f   = '*';
        }
        return sz1 + sz2 + sz3 + n;
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool is_reference_or_pointer_to_function_or_array() const
    {
        return __right_->is_function();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) &&
               __right_->fix_forward_references(t_begin, t_end);
    }
};

class __decltype_node
    : public __node
{
public:

    explicit __decltype_node(__node* expr)
    {
        __right_ = expr;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = 10 + __right_->size();
        return __cached_size_;
    }

    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "decltype(", 9);
        buf += 9;
        buf = __right_->get_demangled_name(buf);
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("decltype()") - 1;
        const ptrdiff_t sz1 = __right_->print(f+std::min(n-1, r), l);
        if (r >= sz1 + n)
        {
            *f++ = 'd';
            *f++ = 'e';
            *f++ = 'c';
            *f++ = 'l';
            *f++ = 't';
            *f++ = 'y';
            *f++ = 'p';
            *f++ = 'e';
            *f   = '(';
            f += 1 + sz1;
            *f   = ')';
        }
        return sz1 + n;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __nested_delimeter
    : public __node
{
public:

    explicit __nested_delimeter(__node* prev, __node* arg)
    {
        __left_ = prev;
        __right_ = arg;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __left_->size() + __right_->size() + 2;
        return __cached_size_;
    }

    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        *buf++ = ':';
        *buf++ = ':';
        return __right_->get_demangled_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("::") - 1;
        const ptrdiff_t sz1 = __left_->print(f, l);
        if (r >= sz1 + n)
        {
            f += sz1;
            *f++ = ':';
            *f++ = ':';
        }
        const ptrdiff_t sz2 = __right_->print(f, l);
        return sz1 + n + sz2;
    }

    virtual bool ends_with_template() const
    {
        return __right_->ends_with_template();
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return __right_->is_ctor_dtor_conv();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end) &&
               __right_->fix_forward_references(t_begin, t_end);
    }
    virtual __node* extract_cv(__node*& rt) const
    {
        return __right_->extract_cv(const_cast<__node*&>(__right_));
    }
};

class __unresolved_name
    : public __node
{
public:

    __unresolved_name(__node* prev, __node* arg)
    {
        __left_ = prev;
        __right_ = arg;
    }

    __unresolved_name(bool global, __node* prev, __node* arg)
    {
        __size_ = global;
        __left_ = prev;
        __right_ = arg;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = (__left_ ? __left_->size() + 2 : 0) +
                                                 __right_->size() + __size_ * 2;
        return __cached_size_;
    }

    virtual char* first_demangled_name(char* buf) const
    {
        if (__size_)
        {
            *buf++ = ':';
            *buf++ = ':';
        }
        if (__left_)
        {
            buf = __left_->get_demangled_name(buf);
            *buf++ = ':';
            *buf++ = ':';
        }
        return __right_->get_demangled_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        ptrdiff_t n = 0;
        if (__size_)
        {
            n = 2;
            if (r >= n)
            {
                f[0] = ':';
                f[1] = ':';
            }
        }
        ptrdiff_t sz1 = 0;
        if (__left_)
        {
            sz1 = __left_->print(f+std::min(n, r), l);
            n += 2;
            if (r >= sz1 + n)
            {
                f[sz1 + n - 2] = ':';
                f[sz1 + n - 1] = ':';
            }
        }
        const ptrdiff_t sz2 = __right_->print(f+std::min(sz1+n, r), l);
        return sz1 + n + sz2;
    }

    virtual bool ends_with_template() const
    {
        return __right_->ends_with_template();
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return __right_->is_ctor_dtor_conv();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        bool r = true;
        if (__left_)
            r = __left_->fix_forward_references(t_begin, t_end);
        return r && __right_->fix_forward_references(t_begin, t_end);
    }
    virtual __node* extract_cv(__node*& rt) const
    {
        return __right_->extract_cv(const_cast<__node*&>(__right_));
    }
};

class __string_literal
    : public __node
{
public:

    virtual size_t first_size() const
    {
        return 14;
    }

    virtual char* first_demangled_name(char* buf) const
    {
        strncpy(buf, "string literal", 14);
        return buf + 14;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = sizeof("string literal") - 1;
        if (r >= n)
        {
            *f++ = 's';
            *f++ = 't';
            *f++ = 'r';
            *f++ = 'i';
            *f++ = 'n';
            *f++ = 'g';
            *f++ = ' ';
            *f++ = 'l';
            *f++ = 'i';
            *f++ = 't';
            *f++ = 'e';
            *f++ = 'r';
            *f++ = 'a';
            *f   = 'l';
        }
        return n;
    }
};

class __constructor
    : public __node
{
public:

    explicit __constructor(__node* name)
    {
        __right_ = name;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __right_->base_size();
        return __cached_size_;
    }

    virtual char* first_demangled_name(char* buf) const
    {
        return __right_->get_base_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        return __right_->print_base_name(f, l);
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool ends_with_template() const
    {
        return __right_->ends_with_template();
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return true;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __destructor
    : public __node
{
public:

    explicit __destructor(__node* name)
    {
        __right_ = name;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
            const_cast<long&>(__cached_size_) = __right_->base_size() + 1;
        return __cached_size_;
    }

    virtual char* first_demangled_name(char* buf) const
    {
        *buf++ = '~';
        return __right_->get_base_name(buf);
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = 1;
        const ptrdiff_t sz = __right_->print_base_name(f+std::min(n, r), l);
        if (r >= n + sz)
            *f = '~';
        return n + sz;
    }
    virtual __node* base_name() const
    {
        return __right_->base_name();
    }
    virtual bool is_ctor_dtor_conv() const
    {
        return true;
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __right_->fix_forward_references(t_begin, t_end);
    }
};

class __dot_suffix
    : public __node
{
public:
    __dot_suffix(__node* name, const char* suffix, unsigned sz)
    {
        __left_ = name;
        __name_ = suffix;
        __size_ = sz;
    }

    virtual size_t first_size() const
    {
        if (__cached_size_ == -1)
        {
            size_t off = __left_->size();
            off += __size_ + 3;
            const_cast<long&>(__cached_size_) = off;
        }
        return __cached_size_;
    }
    virtual char* first_demangled_name(char* buf) const
    {
        buf = __left_->get_demangled_name(buf);
        *buf++ = ' ';
        *buf++ = '(';
        strncpy(buf, __name_, __size_);
        buf += __size_;
        *buf++ = ')';
        return buf;
    }
    virtual ptrdiff_t print_first(char* f, char* l) const
    {
        const ptrdiff_t r = l - f;
        const ptrdiff_t n = 3 + __size_;
        const ptrdiff_t sz = __left_->print(f, l);
        if (r >= n + sz)
        {
            f += sz;
            *f++ = ' ';
            *f++ = '(';
            strncpy(f, __name_, __size_);
            f += __size_;
            *f   = ')';
        }
        return n + sz;
    }
    virtual __node* base_name() const
    {
        return __left_->base_name();
    }
    virtual bool fix_forward_references(__node** t_begin, __node** t_end)
    {
        return __left_->fix_forward_references(t_begin, t_end);
    }
};


enum {invalid_args = -3, invalid_mangled_name, memory_alloc_failure, success,
      not_yet_implemented};

__demangle_tree::__demangle_tree(const char* mangled_name, char* buf, size_t bs)
    : __mangled_name_begin_(0), __mangled_name_end_(0),
      __status_(invalid_mangled_name), __root_(0),
      __node_begin_(0), __node_end_(0), __node_cap_(0),
      __sub_begin_(0), __sub_end_(0), __sub_cap_(0),
      __t_begin_(0), __t_end_(0), __t_cap_(0),
      __tag_templates_(true),
      __fix_forward_references_(false)
{
    size_t n = strlen(mangled_name);
    size_t ms = n + 2*n*sizeof(__node) + 2*n*sizeof(__node*);
    char* m;
    if (ms <= bs)
    {
        m = buf;
        __owns_buf_ = false;
    }
    else
    {
        m = static_cast<char*>(malloc(ms));
        __owns_buf_ = true;
    }
    if (m == NULL)
    {
        __status_ = memory_alloc_failure;
        return;
    }
    __node_begin_ = __node_end_ = (__node*)(m);
    __node_cap_ = __node_begin_ + 2*n;
    __sub_begin_ =  __sub_end_ = (__node**)(__node_cap_);
    __sub_cap_ = __sub_begin_ + n;
    __t_begin_ =  __t_end_ = (__node**)(__sub_cap_);
    __t_cap_ = __t_begin_ + n;
    __mangled_name_begin_ = (const char*)(__t_cap_);
    __mangled_name_end_ = __mangled_name_begin_ + n;
    strncpy(const_cast<char*>(__mangled_name_begin_), mangled_name, n);
}

__demangle_tree::~__demangle_tree()
{
    if (__owns_buf_)
        free(__node_begin_);
}

__demangle_tree::__demangle_tree(__demangle_tree& t)
    : __mangled_name_begin_(t.__mangled_name_begin_),
      __mangled_name_end_(t.__mangled_name_end_),
      __status_(t.__status_), __root_(t.__root_),
      __node_begin_(t.__node_begin_), __node_end_(t.__node_end_),
      __node_cap_(t.__node_cap_),
      __sub_begin_(t.__sub_begin_), __sub_end_(t.__sub_end_),
      __sub_cap_(t.__sub_cap_),
      __t_begin_(t.__t_begin_), __t_end_(t.__t_end_),
      __t_cap_(t.__t_cap_),
      __tag_templates_(t.__tag_templates_),
      __fix_forward_references_(t.__fix_forward_references_),
      __owns_buf_(t.__owns_buf_)
{
    t.__mangled_name_begin_ = 0;
    t.__mangled_name_end_ = 0;
    t.__status_ = invalid_mangled_name;
    t.__root_ = 0;
    t.__node_begin_ = t.__node_end_ = t.__node_cap_ = 0;
    t.__sub_begin_ = t.__sub_end_ = t.__sub_cap_ = 0;
    t.__t_begin_ = t.__t_end_ = t.__t_cap_ = 0;
    t.__owns_buf_ = false;
}

__demangle_tree::__demangle_tree(__demangle_tree_rv rv)
    : __mangled_name_begin_(rv.ptr_->__mangled_name_begin_),
      __mangled_name_end_(rv.ptr_->__mangled_name_end_),
      __status_(rv.ptr_->__status_), __root_(rv.ptr_->__root_),
      __node_begin_(rv.ptr_->__node_begin_), __node_end_(rv.ptr_->__node_end_),
      __node_cap_(rv.ptr_->__node_cap_),
      __sub_begin_(rv.ptr_->__sub_begin_), __sub_end_(rv.ptr_->__sub_end_),
      __sub_cap_(rv.ptr_->__sub_cap_),
      __t_begin_(rv.ptr_->__t_begin_), __t_end_(rv.ptr_->__t_end_),
      __t_cap_(rv.ptr_->__t_cap_),
      __tag_templates_(rv.ptr_->__tag_templates_),
      __fix_forward_references_(rv.ptr_->__fix_forward_references_),
      __owns_buf_(rv.ptr_->__owns_buf_)
{
    rv.ptr_->__mangled_name_begin_ = 0;
    rv.ptr_->__mangled_name_end_ = 0;
    rv.ptr_->__status_ = invalid_mangled_name;
    rv.ptr_->__root_ = 0;
    rv.ptr_->__node_begin_ = rv.ptr_->__node_end_ = rv.ptr_->__node_cap_ = 0;
    rv.ptr_->__sub_begin_ = rv.ptr_->__sub_end_ = rv.ptr_->__sub_cap_ = 0;
    rv.ptr_->__t_begin_ = rv.ptr_->__t_end_ = rv.ptr_->__t_cap_ = 0;
    rv.ptr_->__owns_buf_ = false;
}

int
__demangle_tree::__status() const
{
    return __status_;
}

size_t
__demangle_tree::size() const
{
    return __status_ == success ? __root_->size() : 0;
}

char*
__demangle_tree::__get_demangled_name(char* buf) const
{
    if (__status_ == success)
        return __root_->get_demangled_name(buf);
    return 0;
}

template <class _Tp>
bool
__demangle_tree::__make()
{
    if (__node_end_ < __node_cap_)
    {
        ::new (__node_end_) _Tp();
        __root_ = __node_end_;
        ++__node_end_;
        return true;
    }
    __status_ = memory_alloc_failure;
    return false;
}

template <class _Tp, class _A0>
bool
__demangle_tree::__make(_A0 __a0)
{
    if (__node_end_ < __node_cap_)
    {
        ::new (__node_end_) _Tp(__a0);
        __root_ = __node_end_;
        ++__node_end_;
        return true;
    }
    __status_ = memory_alloc_failure;
    return false;
}

template <class _Tp, class _A0, class _A1>
bool
__demangle_tree::__make(_A0 __a0, _A1 __a1)
{
    if (__node_end_ < __node_cap_)
    {
        ::new (__node_end_) _Tp(__a0, __a1);
        __root_ = __node_end_;
        ++__node_end_;
        return true;
    }
    __status_ = memory_alloc_failure;
    return false;
}

template <class _Tp, class _A0, class _A1, class _A2>
bool
__demangle_tree::__make(_A0 __a0, _A1 __a1, _A2 __a2)
{
    if (__node_end_ < __node_cap_)
    {
        ::new (__node_end_) _Tp(__a0, __a1, __a2);
        __root_ = __node_end_;
        ++__node_end_;
        return true;
    }
    __status_ = memory_alloc_failure;
    return false;
}

template <class _Tp, class _A0, class _A1, class _A2, class _A3>
bool
__demangle_tree::__make(_A0 __a0, _A1 __a1, _A2 __a2, _A3 __a3)
{
    if (__node_end_ < __node_cap_)
    {
        ::new (__node_end_) _Tp(__a0, __a1, __a2, __a3);
        __root_ = __node_end_;
        ++__node_end_;
        return true;
    }
    __status_ = memory_alloc_failure;
    return false;
}

template <class _Tp, class _A0, class _A1, class _A2, class _A3, class _A4>
bool
__demangle_tree::__make(_A0 __a0, _A1 __a1, _A2 __a2, _A3 __a3, _A4 __a4)
{
    if (__node_end_ < __node_cap_)
    {
        ::new (__node_end_) _Tp(__a0, __a1, __a2, __a3, __a4);
        __root_ = __node_end_;
        ++__node_end_;
        return true;
    }
    __status_ = memory_alloc_failure;
    return false;
}

template <class _Tp, class _A0, class _A1, class _A2, class _A3, class _A4,
                     class _A5>
bool
__demangle_tree::__make(_A0 __a0, _A1 __a1, _A2 __a2, _A3 __a3, _A4 __a4,
                        _A5 __a5)
{
    if (__node_end_ < __node_cap_)
    {
        ::new (__node_end_) _Tp(__a0, __a1, __a2, __a3, __a4, __a5);
        __root_ = __node_end_;
        ++__node_end_;
        return true;
    }
    __status_ = memory_alloc_failure;
    return false;
}

// <CV-qualifiers> ::= [r] [V] [K]  # restrict (C99), volatile, const
//                         [R | O]  # & or &&

const char*
__demangle_tree::__parse_cv_qualifiers(const char* first, const char* last,
                                       unsigned& cv, bool look_for_ref_quals)
{
    if (look_for_ref_quals)
    {
        for (; first != last; ++first)
        {
            switch (*first)
            {
            case 'r':
                cv |= 4;
                break;
            case 'V':
                cv |= 2;
                break;
            case 'K':
                cv |= 1;
                break;
            case 'R':
                cv |= 8;
                break;
            case 'O':
                cv |= 16;
                break;
            default:
                return first;
            }
        }
    }
    else
    {
        for (; first != last; ++first)
        {
            switch (*first)
            {
            case 'r':
                cv |= 4;
                break;
            case 'V':
                cv |= 2;
                break;
            case 'K':
                cv |= 1;
                break;
            default:
                return first;
            }
        }
    }
    return first;
}

// <builtin-type> ::= v    # void
//                ::= w    # wchar_t
//                ::= b    # bool
//                ::= c    # char
//                ::= a    # signed char
//                ::= h    # unsigned char
//                ::= s    # short
//                ::= t    # unsigned short
//                ::= i    # int
//                ::= j    # unsigned int
//                ::= l    # long
//                ::= m    # unsigned long
//                ::= x    # long long, __int64
//                ::= y    # unsigned long long, __int64
//                ::= n    # __int128
//                ::= o    # unsigned __int128
//                ::= f    # float
//                ::= d    # double
//                ::= e    # long double, __float80
//                ::= g    # __float128
//                ::= z    # ellipsis
//                ::= Dd   # IEEE 754r decimal floating point (64 bits)
//                ::= De   # IEEE 754r decimal floating point (128 bits)
//                ::= Df   # IEEE 754r decimal floating point (32 bits)
//                ::= Dh   # IEEE 754r half-precision floating point (16 bits)
//                ::= Di   # char32_t
//                ::= Ds   # char16_t
//                ::= Da   # auto (in dependent new-expressions)
//                ::= Dn   # std::nullptr_t (i.e., decltype(nullptr))
//                ::= u <source-name>    # vendor extended type

const char*
__demangle_tree::__parse_builtin_type(const char* first, const char* last)
{
    if (first != last)
    {
        switch (*first)
        {
        case 'v':
            if (__make<__void>())
                ++first;
            break;
        case 'w':
            if (__make<__wchar_t>())
                ++first;
            break;
        case 'b':
            if (__make<__bool>())
                ++first;
            break;
        case 'c':
            if (__make<__char>())
                ++first;
            break;
        case 'a':
            if (__make<__signed_char>())
                ++first;
            break;
        case 'h':
            if (__make<__unsigned_char>())
                ++first;
            break;
        case 's':
            if (__make<__short>())
                ++first;
            break;
        case 't':
            if (__make<__unsigned_short>())
                ++first;
            break;
        case 'i':
            if (__make<__int>())
                ++first;
            break;
        case 'j':
            if (__make<__unsigned_int>())
                ++first;
            break;
        case 'l':
            if (__make<__long>())
                ++first;
            break;
        case 'm':
            if (__make<__unsigned_long>())
                ++first;
            break;
        case 'x':
            if (__make<__long_long>())
                ++first;
            break;
        case 'y':
            if (__make<__unsigned_long_long>())
                ++first;
            break;
        case 'n':
            if (__make<__int128>())
                ++first;
            break;
        case 'o':
            if (__make<__unsigned_int128>())
                ++first;
            break;
        case 'f':
            if (__make<__float>())
                ++first;
            break;
        case 'd':
            if (__make<__double>())
                ++first;
            break;
        case 'e':
            if (__make<__long_double>())
                ++first;
            break;
        case 'g':
            if (__make<__float128>())
                ++first;
            break;
        case 'z':
            if (__make<__ellipsis>())
                ++first;
            break;
        case 'D':
            if (first+1 != last)
            {
                switch (first[1])
                {
                case 'd':
                    if (__make<__decimal64>())
                        first += 2;
                    break;
                case 'e':
                    if (__make<__decimal128>())
                        first += 2;
                    break;
                case 'f':
                    if (__make<__decimal32>())
                        first += 2;
                    break;
                case 'h':
                    if (__make<__decimal16>())
                        first += 2;
                    break;
                case 'i':
                    if (__make<__d_char32_t>())
                        first += 2;
                    break;
                case 's':
                    if (__make<__d_char16_t>())
                        first += 2;
                    break;
                case 'a':
                    if (__make<__auto>())
                        first += 2;
                    break;
                case 'n':
                    if (__make<__nullptr_t>())
                        first += 2;
                    break;
                }
            }
            break;
        }
    }
    return first;
}

// <bare-function-type> ::= <signature type>+
//                      # types are possible return type, then parameter types

const char*
__demangle_tree::__parse_bare_function_type(const char* first, const char* last)
{
    if (first != last)
    {
        __tag_templates_ = false;
        const char* t = __parse_type(first, last);
        if (t != first && __make<__list>(__root_))
        {
            const char* t0 = t;
            __node* head = __root_;
            __node* prev = head;
            while (true)
            {
                t = __parse_type(t0, last);
                if (t != t0)
                {
                    if (__make<__list>(__root_))
                    {
                        t0 = t;
                        prev->__right_ = __root_;
                        __root_->__size_ = prev->__size_ + 1;
                        prev = __root_;
                    }
                    else
                        break;
                }
                else
                {
                    first = t;
                    __root_ = head;
                    break;
                }
            }
        }
        __tag_templates_ = true;
    }
    return first;
}

// <function-type> ::= F [Y] <bare-function-type> E

const char*
__demangle_tree::__parse_function_type(const char* first, const char* last)
{
    if (first != last && *first == 'F')
    {
        const char* t = first+1;
        if (t != last)
        {
            bool externC = false;
            if (*t == 'Y')
            {
                externC = true;
                if (++t == last)
                    return first;
            }
            const char* t1 = __parse_type(t, last);
            if (t1 != t)
            {
                __node* ret = __root_;
                t = t1;
                t1 = __parse_bare_function_type(t, last);
                if (t1 != t && t1 != last && *t1 == 'E')
                {
                    if (dynamic_cast<__void*>(__root_->__left_) != NULL)
                        __root_->__left_ = NULL;
                    if (__make<__function_signature>(ret, __root_))
                    {
                        if (__make<__function>((__node*)0, __root_))
                            first = t1+1;
                    }
                }
            }
        }
    }
    return first;
}

const char*
__demangle_tree::__parse_hex_number(const char* first, const char* last, unsigned long long& n)
{
    const char* t = first;
    for (; t != last && isxdigit(*t); ++t)
    {
        if (t == first)
            n = 0;
        if (isdigit(*t))
            n = n * 16 + *t - '0';
        else if (isupper(*t))
            n = n * 16 + *t - 'A' + 10;
        else
            n = n * 16 + *t - 'a' + 10;
    }
    first = t;
    return first;
}

// <expr-primary> ::= L <type> <value number> E                          # integer literal
//                ::= L <type> <value float> E                           # floating literal
//                ::= L <string type> E                                  # string literal
//                ::= L <nullptr type> E                                 # nullptr literal (i.e., "LDnE")
//                ::= L <type> <real-part float> _ <imag-part float> E   # complex floating point literal (C 2000)
//                ::= L <mangled-name> E                                 # external name

const char*
__demangle_tree::__parse_expr_primary(const char* first, const char* last)
{
    if (last - first >= 4 && *first == 'L')
    {
        switch (first[1])
        {
        case 'w':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__wchar_t_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'b':
            if (first[3] == 'E')
            {
                switch (first[2])
                {
                case '0':
                    if (__make<__bool_literal>("false", 5))
                        first += 4;
                    break;
                case '1':
                    if (__make<__bool_literal>("true", 4))
                        first += 4;
                    break;
                }
            }
            break;
        case 'c':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__char_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'a':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__signed_char_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'h':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__unsigned_char_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 's':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__short_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 't':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__unsigned_short_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'i':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__int_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'j':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__unsigned_int_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'l':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__long_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'm':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__unsigned_long_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'x':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__long_long_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'y':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__unsigned_long_long_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'n':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__int128_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'o':
            {
                const char* t = __parse_number(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__unsigned_int128_literal>(first+2, t))
                        first = t+1;
                }
            }
            break;
        case 'f':
            {
                if (last - (first+2) <= 8)
                    return first;
                unsigned long long j;
                const char* t = __parse_hex_number(first+2, first+10, j);
                if (t != first+2 && t != last && *t == 'E')
                {
                    unsigned i = static_cast<unsigned>(j);
                    float value = *(float*)&i;
                    if (__make<__float_literal>(value))
                        first = t+1;
                }
            }
            break;
        case 'd':
            {
                if (last - (first+2) <= 16)
                    return first;
                unsigned long long j;
                const char* t = __parse_hex_number(first+2, first+18, j);
                if (t != first+2 && t != last && *t == 'E')
                {
                    double value = *(double*)&j;
                    if (__make<__double_literal>(value))
                        first = t+1;
                }
            }
            break;
        case 'e':
            break;
        case '_':
            if (first[2] == 'Z')
            {
                const char* t = __parse_encoding(first+3, last);
                if (t != first+3 && t != last && *t == 'E')
                    first = t+1;
            }
            break;
        default:
            {
                // might be named type
                const char* t = __parse_type(first+1, last);
                if (t != first+1 && t != last)
                {
                    if (*t != 'E')
                    {
                        const char* n = t;
                        for (; n != last && isdigit(*n); ++n)
                            ;
                        if (n != t && n != last && *n == 'E')
                        {
                            if (__make<__cast_literal>(__root_, t, n))
                            {
                                first = n+1;
                                break;
                            }
                        }
                    }
                    else
                    {
                        first = t+1;
                        break;
                    }
                }
            }
            assert(!"case in __parse_expr_primary not implemented");
        }
    }
    return first;
}

const char*
__demangle_tree::__parse_unnamed_type_name(const char* first, const char* last)
{
    if (first != last && *first == 'U')
    {
        assert(!"__parse_unnamed_type_name not implemented");
    }
    return first;
}

// <ctor-dtor-name> ::= C1    # complete object constructor
//                  ::= C2    # base object constructor
//                  ::= C3    # complete object allocating constructor
//                  ::= D0    # deleting destructor
//                  ::= D1    # complete object destructor
//                  ::= D2    # base object destructor

const char*
__demangle_tree::__parse_ctor_dtor_name(const char* first, const char* last)
{
    if (last-first >= 2)
    {
        switch (first[0])
        {
        case 'C':
            switch (first[1])
            {
            case '1':
            case '2':
            case '3':
                if (__make<__constructor>(__root_->base_name()))
                    first += 2;
                break;
            }
            break;
        case 'D':
            switch (first[1])
            {
            case '0':
            case '1':
            case '2':
                if (__make<__destructor>(__root_->base_name()))
                    first += 2;
                break;
            }
            break;
        }
    }
    return first;
}

const char*
__demangle_tree::__parse_unscoped_template_name(const char* first, const char* last)
{
    assert(!"__parse_unscoped_template_name not implemented");
}

// <discriminator> := _ <non-negative number>      # when number < 10
//                 := __ <non-negative number> _   # when number >= 10
//  extension      := decimal-digit+

const char*
__demangle_tree::__parse_discriminator(const char* first, const char* last)
{
    // parse but ignore discriminator
    if (first != last)
    {
        if (*first == '_')
        {
            const char* t1 = first+1;
            if (t1 != last)
            {
                if (isdigit(*t1))
                    first = t1+1;
                else if (*t1 == '_')
                {
                    for (++t1; t1 != last && isdigit(*t1); ++t1)
                        ;
                    if (t1 != last && *t1 == '_')
                        first = t1 + 1;
                }
            }
        }
        else if (isdigit(*first))
        {
            const char* t1 = first+1;
            for (; t1 != last && isdigit(*t1); ++t1)
                ;
            first = t1;
        }
    }
    return first;
}

// <local-name> := Z <function encoding> E <entity name> [<discriminator>]
//              := Z <function encoding> E s [<discriminator>]
//              := Z <function encoding> Ed [ <parameter number> ] _ <entity name>

const char*
__demangle_tree::__parse_local_name(const char* first, const char* last)
{
    if (first != last && *first == 'Z')
    {
        const char* t = __parse_encoding(first+1, last);
        if (t != first+1 && t != last && *t == 'E' && ++t != last)
        {
            __node* encoding = __root_;
            switch (*t)
            {
            case 's':
                {
                    const char*t1 = __parse_discriminator(t+1, last);
                    if (__make<__string_literal>())
                    {
                        if (__make<__nested_delimeter>(encoding, __root_))
                            first = t1;
                    }
                }
                break;
            case 'd':
                assert(!"__parse_local_name d not implemented");
                break;
            default:
                {
                    const char*t1 = __parse_name(t, last);
                    if (t1 != t)
                    {
                        // parse but ignore discriminator
                        t1 = __parse_discriminator(t1, last);
                        if (__make<__nested_delimeter>(encoding, __root_))
                            first = t1;
                    }
                }
                break;
            }
        }
    }
    return first;
}

// <destructor-name> ::= <unresolved-type>                               # e.g., ~T or ~decltype(f())
//                   ::= <simple-id>                                     # e.g., ~A<2*N>

const char*
__demangle_tree::__parse_destructor_name(const char* first, const char* last)
{
    if (first != last)
    {
        const char* t = __parse_unresolved_type(first, last);
        if (t == first)
            t = __parse_simple_id(first, last);
        if (t != first && __make<__destructor>(__root_))
            first = t;
    }
    return first;
}

// <simple-id> ::= <source-name> [ <template-args> ]

const char*
__demangle_tree::__parse_simple_id(const char* first, const char* last)
{
    if (first != last)
    {
        const char* t = __parse_source_name(first, last);
        if (t != first)
            first = __parse_template_args(t, last);
        else
            first = t;
    }
    return first;
}

// <base-unresolved-name> ::= <simple-id>                                # unresolved name
//          extension     ::= <operator-name>                            # unresolved operator-function-id
//          extension     ::= <operator-name> <template-args>            # unresolved operator template-id
//                        ::= on <operator-name>                         # unresolved operator-function-id
//                        ::= on <operator-name> <template-args>         # unresolved operator template-id
//                        ::= dn <destructor-name>                       # destructor or pseudo-destructor;
//                                                                         # e.g. ~X or ~X<N-1>

const char*
__demangle_tree::__parse_base_unresolved_name(const char* first, const char* last)
{
    if (last - first >= 2)
    {
        if ((first[0] == 'o' || first[0] == 'd') && first[1] == 'n')
        {
            if (first[0] == 'o')
            {
                const char* t = __parse_operator_name(first+2, last);
                if (t != first+2)
                    first = __parse_template_args(t, last);
                else
                    first = t;
            }
            else
            {
                const char* t = __parse_destructor_name(first+2, last);
                if (t != first+2)
                    first = t;
            }
        }
        else
        {
            const char* t = __parse_simple_id(first, last);
            if (t == first)
            {
                t = __parse_operator_name(first, last);
                if (t != first)
                    t = __parse_template_args(t, last);
            }
            if (t != first)
                first = t;
        }
    }
    return first;
}

// <unresolved-type> ::= <template-param>
//                   ::= <decltype>
//                   ::= <substitution>

const char*
__demangle_tree::__parse_unresolved_type(const char* first, const char* last)
{
    if (first != last)
    {
        const char* t;
        switch (*first)
        {
        case 'T':
            t = __parse_template_param(first, last);
            if (t != first)
            {
                if (__sub_end_ == __sub_cap_)
                    __status_ = memory_alloc_failure;
                else 
                {
                    *__sub_end_++ = __root_;
                    first = t;
                }
            }
            break;
        case 'D':
            t = __parse_decltype(first, last);
            if (t != first)
            {
                if (__sub_end_ == __sub_cap_)
                    __status_ = memory_alloc_failure;
                else 
                {
                    *__sub_end_++ = __root_;
                    first = t;
                }
            }
            break;
        case 'S':
            t = __parse_substitution(first, last);
            if (t != first)
                first = t;
            break;
        }
    }
    return first;
}

// <unresolved-qualifier-level> ::= <source-name> [ <template-args> ]

const char*
__demangle_tree::__parse_unresolved_qualifier_level(const char* first, const char* last)
{
    if (first != last)
    {
            const char* t = __parse_source_name(first, last);
            if (t != first)
                first = __parse_template_args(t, last);
    }
    return first;
}

// <unresolved-name>
//  extension        ::= srN <unresolved-type> [<template-args>] <unresolved-qualifier-level>* E <base-unresolved-name>
//                   ::= [gs] <base-unresolved-name>                     # x or (with "gs") ::x
//                   ::= [gs] sr <unresolved-qualifier-level>+ E <base-unresolved-name>  
//                                                                       # A::x, N::y, A<T>::z; "gs" means leading "::"
//                   ::= sr <unresolved-type> <base-unresolved-name>     # T::x / decltype(p)::x
//                                                                       # T::N::x /decltype(p)::N::x
//  (ignored)        ::= srN <unresolved-type>  <unresolved-qualifier-level>+ E <base-unresolved-name>

const char*
__demangle_tree::__parse_unresolved_name(const char* first, const char* last)
{
    if (last - first > 2)
    {
        const char* t = first;
        bool global = false;
        if (t[0] == 'g' && t[1] == 's')
        {
            global = true;
            t += 2;
        }
        const char* t2 = __parse_base_unresolved_name(t, last);
        if (t2 != t)
        {
            if (__make<__unresolved_name>(global, (__node*)0, __root_))
                first = t2;
        }
        else if (last - t > 2 && t[0] == 's' && t[1] == 'r')
        {
            if (!global && t[2] == 'N')
            {
                t2 = __parse_unresolved_type(t+3, last);
                if (t2 != t+3 && t2 != last)
                {
                    t = __parse_template_args(t2, last);
                    if (t == last)
                        return first;
                    __node* name = __root_;
                    while (*t != 'E')
                    {
                        t2 = __parse_unresolved_qualifier_level(t, last);
                        if (t2 == t || t2 == last)
                            return first;
                        if (!__make<__nested_delimeter>(name, __root_))
                            return first;
                        name = __root_;
                        t = t2;
                    }
                    t2 = __parse_base_unresolved_name(++t, last);
                    if (t2 != t && __make<__unresolved_name>(false, name, __root_))
                        first = t2;
                }
            }
            else
            {
                if (!global)
                {
                    t2 = __parse_unresolved_type(t+2, last);
                    if (t2 != t+2)
                    {
                        t = t2;
                        __node* name = __root_;
                        t2 = __parse_base_unresolved_name(t, last);
                        if (t2 != t && __make<__unresolved_name>(false, name, __root_))
                            return t2;
                        return first;
                    }
                }
                t2 = __parse_unresolved_qualifier_level(t+2, last);
                if (t2 != t+2 && t2 != last)
                {
                    __node* name = __root_;
                    t = t2;
                    while (*t != 'E')
                    {
                        t2 = __parse_unresolved_qualifier_level(t, last);
                        if (t2 == t || t2 == last)
                            return first;
                        if (!__make<__nested_delimeter>(name, __root_))
                            return first;
                        name = __root_;
                        t = t2;
                    }
                    t2 = __parse_base_unresolved_name(++t, last);
                    if (t2 != t && __make<__unresolved_name>(global, name, __root_))
                        first = t2;
                }
            }
        }
    }
    return first;
}

// <function-param> ::= fp <top-level CV-qualifiers> _                                     # L == 0, first parameter
//                  ::= fp <top-level CV-qualifiers> <parameter-2 non-negative number> _   # L == 0, second and later parameters
//                  ::= fL <L-1 non-negative number> p <top-level CV-qualifiers> _         # L > 0, first parameter
//                  ::= fL <L-1 non-negative number> p <top-level CV-qualifiers>

const char*
__demangle_tree::__parse_function_param(const char* first, const char* last)
{
    if (last - first >= 3 && *first == 'f')
    {
        if (first[1] == 'p')
        {
            assert(!"__parse_function_param not implemented");
        }
        else if (first[1] == 'L')
        {
            assert(!"__parse_function_param not implemented");
        }
    }
    return first;
}

// at <type>                                            # alignof (a type)

const char*
__demangle_tree::__parse_alignof_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 'a' && first[1] == 't')
    {
        const char* t = __parse_type(first+2, last);
        if (t != first+2)
        {
            if (__make<__operator_alignof_expression>(__root_))
                first = t;
        }
    }
    return first;
}

// cc <type> <expression>                               # const_cast<type> (expression)

const char*
__demangle_tree::__parse_const_cast_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 'c' && first[1] == 'c')
    {
        const char* t = __parse_type(first+2, last);
        if (t != first+2)
        {
            __node* type = __root_;
            const char* t1 = __parse_expression(t, last);
            if (t1 != t)
            {
                if (__make<__const_cast>(type, __root_))
                    first = t1;
            }
        }
    }
    return first;
}

// cl <expression>+ E                                   # call

const char*
__demangle_tree::__parse_call_expr(const char* first, const char* last)
{
    if (last - first >= 4 && first[0] == 'c' && first[1] == 'l')
    {
        const char* t = __parse_expression(first+2, last);
        if (t != first+2)
        {
            if (t == last)
                return first;
            __node* name = __root_;
            __node* args = 0;
            __node* prev = 0;
            while (*t != 'E')
            {
                const char* t1 = __parse_expression(t, last);
                if (t1 == t || t1 == last)
                    return first;
                if (!__make<__list>(__root_))
                    return first;
                if (args == 0)
                    args = __root_;
                if (prev)
                {
                    prev->__right_ = __root_;
                    __root_->__size_ = prev->__size_ + 1;
                }
                prev = __root_;
                t = t1;
            }
            ++t;
            if (__make<__call_expr>(name, args))
                first = t;
        }
    }
    return first;
}

// cv <type> <expression>                               # conversion with one argument
// cv <type> _ <expression>* E                          # conversion with a different number of arguments

const char*
__demangle_tree::__parse_conversion_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 'c' && first[1] == 'v')
    {
        const char* t = __parse_type(first+2, last);
        if (t != first+2 && t != last)
        {
            __node* type = __root_;
            __node* args = 0;
            if (*t != '_')
            {
                const char* t1 = __parse_expression(t, last);
                if (t1 == t)
                    return first;
                args = __root_;
                t = t1;
            }
            else
            {
                ++t;
                if (t == last)
                    return first;
                __node* prev = 0;
                while (*t != 'E')
                {
                    const char* t1 = __parse_expression(t, last);
                    if (t1 == t || t1 == last)
                        return first;
                    if (!__make<__list>(__root_))
                        return first;
                    if (args == 0)
                        args = __root_;
                    if (prev)
                    {
                        prev->__right_ = __root_;
                        __root_->__size_ = prev->__size_ + 1;
                    }
                    prev = __root_;
                    t = t1;
                }
                ++t;
            }
            if (__make<__operator_cast>(type, args))
                first = t;
        }
    }
    return first;
}

// [gs] da <expression>                                 # delete[] expression

const char*
__demangle_tree::__parse_delete_array_expr(const char* first, const char* last)
{
    if (last - first >= 4)
    {
        const char* t = first;
        bool parsed_gs = false;
        if (t[0] == 'g' && t[1] == 's')
        {
            t += 2;
            parsed_gs = true;
        }
        if (t[0] == 'd' && t[1] == 'a')
        {
            t += 2;
            const char* t1 = __parse_expression(t, last);
            if (t1 != t)
            {
                if (__make<__delete_array_expr>(parsed_gs, __root_))
                    first = t1;
            }
        }
    }
    return first;
}

// dc <type> <expression>                               # dynamic_cast<type> (expression)

const char*
__demangle_tree::__parse_dynamic_cast_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 'd' && first[1] == 'c')
    {
        const char* t = __parse_type(first+2, last);
        if (t != first+2)
        {
            __node* type = __root_;
            const char* t1 = __parse_expression(t, last);
            if (t1 != t)
            {
                if (__make<__dynamic_cast>(type, __root_))
                    first = t1;
            }
        }
    }
    return first;
}

// [gs] dl <expression>                                 # delete expression

const char*
__demangle_tree::__parse_delete_expr(const char* first, const char* last)
{
    if (last - first >= 4)
    {
        const char* t = first;
        bool parsed_gs = false;
        if (t[0] == 'g' && t[1] == 's')
        {
            t += 2;
            parsed_gs = true;
        }
        if (t[0] == 'd' && t[1] == 'l')
        {
            t += 2;
            const char* t1 = __parse_expression(t, last);
            if (t1 != t)
            {
                if (__make<__delete_expr>(parsed_gs, __root_))
                    first = t1;
            }
        }
    }
    return first;
}

// ds <expression> <expression>                         # expr.*expr

const char*
__demangle_tree::__parse_dot_star_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 'd' && first[1] == 's')
    {
        const char* t = __parse_expression(first+2, last);
        if (t != first+2)
        {
            __node* expr = __root_;
            const char* t1 = __parse_expression(t, last);
            if (t1 != t)
            {
                if (__make<__dot_star_expr>(expr, __root_))
                    first = t1;
            }
        }
    }
    return first;
}

// dt <expression> <unresolved-name>                    # expr.name

const char*
__demangle_tree::__parse_dot_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 'd' && first[1] == 't')
    {
        const char* t = __parse_expression(first+2, last);
        if (t != first+2)
        {
            __node* expr = __root_;
            const char* t1 = __parse_unresolved_name(t, last);
            if (t1 != t)
            {
                if (__make<__dot_expr>(expr, __root_))
                    first = t1;
            }
        }
    }
    return first;
}

// mm_ <expression>                                     # prefix --

const char*
__demangle_tree::__parse_decrement_expr(const char* first, const char* last)
{
    if (last - first > 3 && first[0] == 'm' && first[1] == 'm' && first[2] == '_')
    {
        const char* t = __parse_expression(first+3, last);
        if (t != first+3)
        {
            if (__make<__operator_decrement>(true, __root_))
                first = t;
        }
    }
    return first;
}

// pp_ <expression>                                     # prefix ++

const char*
__demangle_tree::__parse_increment_expr(const char* first, const char* last)
{
    if (last - first > 3 && first[0] == 'p' && first[1] == 'p' && first[2] == '_')
    {
        const char* t = __parse_expression(first+3, last);
        if (t != first+3)
        {
            if (__make<__operator_increment>(true, __root_))
                first = t;
        }
    }
    return first;
}

// [gs] nw <expression>* _ <type> E                     # new (expr-list) type
// [gs] nw <expression>* _ <type> <initializer>         # new (expr-list) type (init)
// [gs] na <expression>* _ <type> E                     # new[] (expr-list) type
// [gs] na <expression>* _ <type> <initializer>         # new[] (expr-list) type (init)
// <initializer> ::= pi <expression>* E                 # parenthesized initialization

const char*
__demangle_tree::__parse_new_expr(const char* first, const char* last)
{
    if (last - first >= 4)
    {
        const char* t = first;
        bool parsed_gs = false;
        if (t[0] == 'g' && t[1] == 's')
        {
            t += 2;
            parsed_gs = true;
        }
        if (t[0] == 'n' && (t[1] == 'w' || t[1] == 'a'))
        {
            bool is_array = t[1] == 'a';
            t += 2;
            if (t == last)
                return first;
            __node* expr = 0;
            __node* prev = 0;
            while (*t != '_')
            {
                const char* t1 = __parse_expression(t, last);
                if (t1 == t || t1 == last)
                    return first;
                if (!__make<__list>(__root_))
                    return first;
                if (expr == 0)
                    expr = __root_;
                if (prev)
                {
                    prev->__right_ = __root_;
                    __root_->__size_ = prev->__size_ + 1;
                }
                prev = __root_;
                t = t1;
            }
            ++t;
            const char* t1 = __parse_type(t, last);
            if (t1 == t || t1 == last)
                return first;
            t = t1;
            __node* type = __root_;
            __node* init = 0;
            prev = 0;
            bool has_init = false;
            if (last - t >= 3 && t[0] == 'p' && t[1] == 'i')
            {
                t += 2;
                has_init = true;
                while (*t != 'E')
                {
                    t1 = __parse_expression(t, last);
                    if (t1 == t || t1 == last)
                        return first;
                    if (!__make<__list>(__root_))
                        return first;
                    if (init == 0)
                        init = __root_;
                    if (prev)
                    {
                        prev->__right_ = __root_;
                        __root_->__size_ = prev->__size_ + 1;
                    }
                    prev = __root_;
                    t = t1;
                }
            }
            if (*t != 'E')
                return first;
            if (__make<__new_expr>(parsed_gs, is_array, has_init,
                                   expr, type, init))
                first = t;
        }
    }
    return first;
}

// pt <expression> <unresolved-name>                    # expr->name

const char*
__demangle_tree::__parse_arrow_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 'p' && first[1] == 't')
    {
        const char* t = __parse_expression(first+2, last);
        if (t != first+2)
        {
            __node* expr = __root_;
            const char* t1 = __parse_unresolved_name(t, last);
            if (t1 != t)
            {
                if (__make<__arrow_expr>(expr, __root_))
                    first = t1;
            }
        }
    }
    return first;
}

// rc <type> <expression>                               # reinterpret_cast<type> (expression)

const char*
__demangle_tree::__parse_reinterpret_cast_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 'r' && first[1] == 'c')
    {
        const char* t = __parse_type(first+2, last);
        if (t != first+2)
        {
            __node* type = __root_;
            const char* t1 = __parse_expression(t, last);
            if (t1 != t)
            {
                if (__make<__reinterpret_cast>(type, __root_))
                    first = t1;
            }
        }
    }
    return first;
}

// sc <type> <expression>                               # static_cast<type> (expression)

const char*
__demangle_tree::__parse_static_cast_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'c')
    {
        const char* t = __parse_type(first+2, last);
        if (t != first+2)
        {
            __node* type = __root_;
            const char* t1 = __parse_expression(t, last);
            if (t1 != t)
            {
                if (__make<__static_cast>(type, __root_))
                    first = t1;
            }
        }
    }
    return first;
}

// st <type>                                            # sizeof (a type)

const char*
__demangle_tree::__parse_sizeof_type_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 't')
    {
        const char* t = __parse_type(first+2, last);
        if (t != first+2)
        {
            if (__make<__operator_sizeof_expression>(__root_))
                first = t;
        }
    }
    return first;
}

// sZ <template-param>                                  # size of a parameter pack

const char*
__demangle_tree::__parse_sizeof_param_pack_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'Z' && first[2] == 'T')
    {
        const char* t = __parse_template_param(first+2, last);
        if (t != first+2)
        {
            if (__make<__operator_sizeof_param_pack>(__root_))
                first = t;
        }
    }
    return first;
}

// sZ <function-param>                                  # size of a function parameter pack

const char*
__demangle_tree::__parse_sizeof_function_param_pack_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'Z' && first[2] == 'f')
    {
        const char* t = __parse_function_param(first+2, last);
        if (t != first+2)
        {
            if (__make<__operator_sizeof_param_pack>(__root_))
                first = t;
        }
    }
    return first;
}

// sp <expression>                                  # pack expansion

const char*
__demangle_tree::__parse_pack_expansion(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'p')
    {
        const char* t = __parse_expression(first+2, last);
        if (t != first+2)
        {
            if (__make<__pack_expansion>(__root_))
                first = t;
        }
    }
    return first;
}

// te <expression>                                      # typeid (expression)
// ti <type>                                            # typeid (type)

const char*
__demangle_tree::__parse_typeid_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 't' && (first[1] == 'e' || first[1] == 'i'))
    {
        const char* t;
        if (first[1] == 'e')
            t = __parse_expression(first+2, last);
        else
            t = __parse_type(first+2, last);
        if (t != first+2)
        {
            if (__make<__typeid>(__root_))
                first = t;
        }
    }
    return first;
}

// tw <expression>                                      # throw expression

const char*
__demangle_tree::__parse_throw_expr(const char* first, const char* last)
{
    if (last - first >= 3 && first[0] == 't' && first[1] == 'w')
    {
        const char* t = __parse_expression(first+2, last);
        if (t != first+2)
        {
            if (__make<__throw>(__root_))
                first = t;
        }
    }
    return first;
}

// <expression> ::= <unary operator-name> <expression>
//              ::= <binary operator-name> <expression> <expression>
//              ::= <ternary operator-name> <expression> <expression> <expression>
//              ::= cl <expression>+ E                                   # call
//              ::= cv <type> <expression>                               # conversion with one argument
//              ::= cv <type> _ <expression>* E                          # conversion with a different number of arguments
//              ::= [gs] nw <expression>* _ <type> E                     # new (expr-list) type
//              ::= [gs] nw <expression>* _ <type> <initializer>         # new (expr-list) type (init)
//              ::= [gs] na <expression>* _ <type> E                     # new[] (expr-list) type
//              ::= [gs] na <expression>* _ <type> <initializer>         # new[] (expr-list) type (init)
//              ::= [gs] dl <expression>                                 # delete expression
//              ::= [gs] da <expression>                                 # delete[] expression
//              ::= pp_ <expression>                                     # prefix ++
//              ::= mm_ <expression>                                     # prefix --
//              ::= ti <type>                                            # typeid (type)
//              ::= te <expression>                                      # typeid (expression)
//              ::= dc <type> <expression>                               # dynamic_cast<type> (expression)
//              ::= sc <type> <expression>                               # static_cast<type> (expression)
//              ::= cc <type> <expression>                               # const_cast<type> (expression)
//              ::= rc <type> <expression>                               # reinterpret_cast<type> (expression)
//              ::= st <type>                                            # sizeof (a type)
//              ::= at <type>                                            # alignof (a type)
//              ::= <template-param>
//              ::= <function-param>
//              ::= dt <expression> <unresolved-name>                    # expr.name
//              ::= pt <expression> <unresolved-name>                    # expr->name
//              ::= ds <expression> <expression>                         # expr.*expr
//              ::= sZ <template-param>                                  # size of a parameter pack
//              ::= sZ <function-param>                                  # size of a function parameter pack
//              ::= sp <expression>                                      # pack expansion
//              ::= tw <expression>                                      # throw expression
//              ::= tr                                                   # throw with no operand (rethrow)
//              ::= <unresolved-name>                                    # f(p), N::f(p), ::f(p),
//                                                                       # freestanding dependent name (e.g., T::x),
//                                                                       # objectless nonstatic member reference
//              ::= <expr-primary>

const char*
__demangle_tree::__parse_expression(const char* first, const char* last)
{
    if (last - first >= 2)
    {
        const char* t = first;
        bool parsed_gs = false;
        if (last - first >= 4 && t[0] == 'g' && t[1] == 's')
        {
            t += 2;
            parsed_gs = true;
        }
        switch (*t)
        {
        case 'L':
            t = __parse_expr_primary(first, last);
            break;
        case 'T':
            t = __parse_template_param(first, last);
            break;
        case 'f':
            t = __parse_function_param(first, last);
            break;
        case 'a':
            if (t[1] == 't')
                t = __parse_alignof_expr(first, last);
            break;
        case 'c':
            switch (t[1])
            {
            case 'c':
                t = __parse_const_cast_expr(first, last);
                break;
            case 'l':
                t = __parse_call_expr(first, last);
                break;
            case 'v':
                t = __parse_conversion_expr(first, last);
                break;
            }
            break;
        case 'd':
            switch (t[1])
            {
            case 'a':
                t = __parse_delete_array_expr(first, last);
                break;
            case 'c':
                t = __parse_dynamic_cast_expr(first, last);
                break;
            case 'l':
                t = __parse_delete_expr(first, last);
                break;
            case 's':
                t = __parse_dot_star_expr(first, last);
                break;
            case 't':
                t = __parse_dot_expr(first, last);
                break;
            }
            break;
        case 'm':
            t = __parse_decrement_expr(first, last);
            break;
        case 'n':
            switch (t[1])
            {
            case 'a':
            case 'w':
                t = __parse_new_expr(first, last);
                break;
            }
            break;
        case 'p':
            switch (t[1])
            {
            case 'p':
                t = __parse_increment_expr(first, last);
                break;
            case 't':
                t = __parse_arrow_expr(first, last);
                break;
            }
            break;
        case 'r':
            t = __parse_reinterpret_cast_expr(first, last);
            break;
        case 's':
            switch (t[1])
            {
            case 'c':
                t = __parse_static_cast_expr(first, last);
                break;
            case 'p':
                t = __parse_pack_expansion(first, last);
                break;
            case 't':
                t = __parse_sizeof_type_expr(first, last);
                break;
            case 'Z':
                if (last - t >= 3)
                {
                    switch (t[2])
                    {
                    case 'T':
                        t = __parse_sizeof_param_pack_expr(first, last);
                        break;
                    case 'f':
                        t = __parse_sizeof_function_param_pack_expr(first, last);
                        break;
                    }
                }
                break;
            }
            break;
        case 't':
            switch (t[1])
            {
            case 'e':
            case 'i':
                t = __parse_typeid_expr(first, last);
                break;
            case 'r':
                if (__make<__rethrow>())
                    t = first +2;
                break;
            case 'w':
                t = __parse_throw_expr(first, last);
                break;
            }
            break;
        }
        if ((!parsed_gs && t == first) || (parsed_gs && t == first+2))
        {
            int op;
            t = __parse_operator_name(first, last, &op);
            if (t == first)
                first = __parse_unresolved_name(first, last);
            else
                first = t;
        }
        else
            first = t;
    }
    return first;
}

// <array-type> ::= A <positive dimension number> _ <element type>
//              ::= A [<dimension expression>] _ <element type>

const char*
__demangle_tree::__parse_array_type(const char* first, const char* last)
{
    if (first != last && *first == 'A' && first+1 != last)
    {
        if (first[1] == '_')
        {
            const char* t = __parse_type(first+2, last);
            if (t != first+2)
            {
                if (__make<__array>(__root_))
                    first = t;
            }
        }
        else if ('1' <= first[1] && first[1] <= '9')
        {
            size_t dim = first[1] - '0';
            const char* t = first+2;
            for (; t != last && isdigit(*t); ++t)
                dim = dim * 10 + *t - '0';
            if (t != last && *t == '_')
            {
                const char* t2 = __parse_type(t+1, last);
                if (t2 != t+1)
                {
                    if (__make<__array>(__root_, dim))
                        first = t2;
                }
            }
        }
        else
        {
            const char* t = __parse_expression(first+1, last);
            if (t != first+1 && t != last && *t == '_')
            {
                __node* dim = __root_;
                const char* t2 = __parse_type(++t, last);
                if (t2 != t)
                {
                    if (__make<__array>(__root_, dim))
                        first = t2;
                }
            }
        }
    }
    return first;
}

// <class-enum-type> ::= <name>

const char*
__demangle_tree::__parse_class_enum_type(const char* first, const char* last)
{
    return __parse_name(first, last);
}

// <pointer-to-member-type> ::= M <class type> <member type>

const char*
__demangle_tree::__parse_pointer_to_member_type(const char* first, const char* last)
{
    if (first != last && *first == 'M')
    {
        const char* t = __parse_type(first+1, last);
        if (t != first+1)
        {
            __node* class_type = __root_;
            const char* t2 = __parse_type(t, last, true, true);
            if (t2 != t)
            {
                if (__make<__pointer_to_member_type>(class_type, __root_))
                    first = t2;
            }
        }
    }
    return first;
}

// <decltype>  ::= Dt <expression> E  # decltype of an id-expression or class member access (C++0x)
//             ::= DT <expression> E  # decltype of an expression (C++0x)

const char*
__demangle_tree::__parse_decltype(const char* first, const char* last)
{
    if (last - first >= 4 && first[0] == 'D')
    {
        switch (first[1])
        {
        case 't':
        case 'T':
            {
                const char* t = __parse_expression(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (__make<__decltype_node>(__root_))
                        first = t+1;
                }
            }
            break;
        }
    }
    return first;
}

// <template-param> ::= T_    # first template parameter
//                  ::= T <parameter-2 non-negative number> _

const char*
__demangle_tree::__parse_template_param(const char* first, const char* last)
{
    if (last - first >= 2)
    {
        if (*first == 'T')
        {
            if (first[1] == '_')
            {
                if (__t_begin_ != __t_end_)
                {
                    if (__make<__sub>(*__t_begin_))
                        first += 2;
                }
                else
                {
                    if (__make<__sub>(size_t(0)))
                    {
                        first += 2;
                        __fix_forward_references_ = true;
                    }
                }
            }
            else if (isdigit(first[1]))
            {
                const char* t = first+1;
                size_t sub = *t - '0';
                for (++t; t != last && isdigit(*t); ++t)
                {
                    sub *= 10;
                    sub += *t - '0';
                }
                if (t == last || *t != '_')
                    return first;
                ++sub;
                if (sub < __t_end_ - __t_begin_)
                {
                    if (__make<__sub>(__t_begin_[sub]))
                        first = t+1;
                }
                else
                {
                    if (__make<__sub>(sub))
                    {
                        first = t+1;
                        __fix_forward_references_ = true;
                    }
                }
            }
        }
    }
    return first;
}

// <type> ::= <builtin-type>
//        ::= <function-type>
//        ::= <class-enum-type>
//        ::= <array-type>
//        ::= <pointer-to-member-type>
//        ::= <template-param>
//        ::= <template-template-param> <template-args>
//        ::= <decltype>
//        ::= <substitution>
//        ::= <CV-qualifiers> <type>
//        ::= P <type>        # pointer-to
//        ::= R <type>        # reference-to
//        ::= O <type>        # rvalue reference-to (C++0x)
//        ::= C <type>        # complex pair (C 2000)
//        ::= G <type>        # imaginary (C 2000)
//        ::= Dp <type>       # pack expansion (C++0x)
//        ::= U <source-name> <type>  # vendor extended type qualifier

const char*
__demangle_tree::__parse_type(const char* first, const char* last,
                              bool try_to_parse_template_args,
                              bool look_for_ref_quals)
{
    unsigned cv = 0;
    const char* t = __parse_cv_qualifiers(first, last, cv, look_for_ref_quals);
    if (t != first)
    {
        const char* t2 = __parse_type(t, last, try_to_parse_template_args);
        if (t2 != t)
        {
            if (__make<__cv_qualifiers>(cv, __root_))
            {
                if (__sub_end_ == __sub_cap_)
                    __status_ = memory_alloc_failure;
                else
                {
                    *__sub_end_++ = __root_;
                    first = t2;
                }
            }
        }
        return first;
    }
    if (first != last)
    {
        switch (*first)
        {
        case 'A':
            t = __parse_array_type(first, last);
            if (t != first)
            {
                if (__sub_end_ == __sub_cap_)
                    __status_ = memory_alloc_failure;
                else
                {
                    *__sub_end_++ = __root_;
                    first = t;
                }
            }
            break;
        case 'C':
            t = __parse_type(first+1, last, try_to_parse_template_args);
            if (t != first+1)
            {
                if (__make<__d_complex>(__root_))
                {
                    if (__sub_end_ == __sub_cap_)
                        __status_ = memory_alloc_failure;
                    else
                    {
                        *__sub_end_++ = __root_;
                        first = t;
                    }
                }
                return first;
            }
            break;
        case 'F':
            t = __parse_function_type(first, last);
            if (t != first)
            {
                if (__sub_end_ == __sub_cap_)
                    __status_ = memory_alloc_failure;
                else
                {
                    *__sub_end_++ = __root_;
                    first = t;
                }
            }
            break;
        case 'G':
            t = __parse_type(first+1, last, try_to_parse_template_args);
            if (t != first+1)
            {
                if (__make<__imaginary>(__root_))
                {
                    if (__sub_end_ == __sub_cap_)
                        __status_ = memory_alloc_failure;
                    else
                    {
                        *__sub_end_++ = __root_;
                        first = t;
                    }
                }
                return first;
            }
            break;
        case 'M':
            t = __parse_pointer_to_member_type(first, last);
            if (t != first)
            {
                if (__sub_end_ == __sub_cap_)
                    __status_ = memory_alloc_failure;
                else
                {
                    *__sub_end_++ = __root_;
                    first = t;
                }
            }
            break;
        case 'O':
            t = __parse_type(first+1, last, try_to_parse_template_args);
            if (t != first+1)
            {
                if (__make<__rvalue_reference_to>(__root_))
                {
                    if (__sub_end_ == __sub_cap_)
                        __status_ = memory_alloc_failure;
                    else
                    {
                        *__sub_end_++ = __root_;
                        first = t;
                    }
                }
                return first;
            }
            break;
        case 'P':
            t = __parse_type(first+1, last, try_to_parse_template_args);
            if (t != first+1)
            {
                if (__make<__pointer_to>(__root_))
                {
                    if (__sub_end_ == __sub_cap_)
                        __status_ = memory_alloc_failure;
                    else
                    {
                        *__sub_end_++ = __root_;
                        first = t;
                    }
                }
                return first;
            }
            break;
        case 'R':
            t = __parse_type(first+1, last, try_to_parse_template_args);
            if (t != first+1)
            {
                if (__make<__lvalue_reference_to>(__root_))
                {
                    if (__sub_end_ == __sub_cap_)
                        __status_ = memory_alloc_failure;
                    else
                    {
                        *__sub_end_++ = __root_;
                        first = t;
                    }
                }
                return first;
            }
            break;
        case 'T':
            t = __parse_template_param(first, last);
            if (t != first)
            {
                if (__sub_end_ == __sub_cap_)
                    __status_ = memory_alloc_failure;
                else 
                {
                    *__sub_end_++ = __root_;
                    if (try_to_parse_template_args)
                    {
                        const char* t2 = __parse_template_args(t, last);
                        if (t2 != t)
                        {
                            if (__sub_end_ < __sub_cap_)
                            {
                                *__sub_end_++ = __root_;
                                first = t2;
                            }
                            else
                                __status_ = memory_alloc_failure;
                        }
                        else
                        {
                            first = t;
                        }
                    }
                    else
                    {
                        first = t;
                    }
                }
            }
            break;
        case 'U':
            if (first+1 != last)
            {
                t = __parse_source_name(first+1, last);
                if (t != first+1)
                {
                    __node*  name = __root_;
                    const char* t2 = __parse_type(t, last, try_to_parse_template_args);
                    if (t2 != t)
                    {
                        if (__make<__extended_qualifier>(name, __root_))
                        {
                            if (__sub_end_ == __sub_cap_)
                                __status_ = memory_alloc_failure;
                            else
                            {
                                *__sub_end_++ = __root_;
                                first = t2;
                            }
                        }
                        return first;
                    }
                }
            }
            break;
        case 'S':
            if (first+1 != last && first[1] == 't')
            {
                t = __parse_class_enum_type(first, last);
                if (t != first)
                {
                    if (__sub_end_ == __sub_cap_)
                        __status_ = memory_alloc_failure;
                    else
                    {
                        *__sub_end_++ = __root_;
                        first = t;
                    }
                }
            }
            else
            {
                t = __parse_substitution(first, last);
                if (t != first)
                {
                    first = t;
                    // Parsed a substitution.  If the substitution is a
                    //  <template-param> it might be followed by <template-args>.
                    t = __parse_template_args(first, last);
                    if (t != first)
                    {
                        // Need to create substitution for <template-template-param> <template-args>
                        if (__sub_end_ == __sub_cap_)
                            __status_ = memory_alloc_failure;
                        else
                        {
                            *__sub_end_++ = __root_;
                            first = t;
                        }
                    }
                }
            }
            break;
        case 'D':
            if (first+1 != last)
            {
                switch (first[1])
                {
                case 'p':
                    t = __parse_type(first+2, last, try_to_parse_template_args);
                    if (t != first+1)
                    {
                        if (__make<__pack_expansion>(__root_))
                        {
                            if (__sub_end_ == __sub_cap_)
                                __status_ = memory_alloc_failure;
                            else
                            {
                                *__sub_end_++ = __root_;
                                first = t;
                            }
                        }
                        return first;
                    }
                    break;
                case 't':
                case 'T':
                    t = __parse_decltype(first, last);
                    if (t != first)
                    {
                       if (__sub_end_ == __sub_cap_)
                            __status_ = memory_alloc_failure;
                        else
                        {
                            *__sub_end_++ = __root_;
                            first = t;
                        }
                        return first;
                    }
                    break;
                }
            }
            // drop through
        default:
            // must check for builtin-types before class-enum-types to avoid
            // ambiguities with operator-names
            t = __parse_builtin_type(first, last);
            if (t != first)
            {
                first = t;
            }
            else
            {
                t = __parse_class_enum_type(first, last);
                if (t != first)
                {
                    if (__sub_end_ == __sub_cap_)
                        __status_ = memory_alloc_failure;
                    else
                    {
                        *__sub_end_++ = __root_;
                        first = t;
                    }
                }
            }
            break;
        }
    }
    return first;
}

// <number> ::= [n] <non-negative decimal integer>

const char*
__demangle_tree::__parse_number(const char* first, const char* last)
{
    if (first != last)
    {
        const char* t = first;
        if (*t == 'n')
            ++t;
        if (t != last)
        {
            if (*t == '0')
            {
                first = t+1;
            }
            else if ('1' <= *t && *t <= '9')
            {
                first = t+1;
                while (first != last && isdigit(*first))
                    ++first;
            }
        }
    }
    return first;
}

// <call-offset> ::= h <nv-offset> _
//               ::= v <v-offset> _
// 
// <nv-offset> ::= <offset number>
//               # non-virtual base override
// 
// <v-offset>  ::= <offset number> _ <virtual offset number>
//               # virtual base override, with vcall offset

const char*
__demangle_tree::__parse_call_offset(const char* first, const char* last)
{
    if (first != last)
    {
        switch (*first)
        {
        case 'h':
            {
            const char* t = __parse_number(first + 1, last);
            if (t != first + 1 && t != last && *t == '_')
                first = t + 1;
            }
            break;
        case 'v':
            {
            const char* t = __parse_number(first + 1, last);
            if (t != first + 1 && t != last && *t == '_')
            {
                const char* t2 = __parse_number(++t, last);
                if (t2 != t && t2 != last && *t2 == '_')
                    first = t2 + 1;
            }
            }
            break;
        }
    }
    return first;
}

// <special-name> ::= TV <type>    # virtual table
//                ::= TT <type>    # VTT structure (construction vtable index)
//                ::= TI <type>    # typeinfo structure
//                ::= TS <type>    # typeinfo name (null-terminated byte string)
//                ::= Tc <call-offset> <call-offset> <base encoding>
//                    # base is the nominal target function of thunk
//                    # first call-offset is 'this' adjustment
//                    # second call-offset is result adjustment
//                ::= T <call-offset> <base encoding>
//                    # base is the nominal target function of thunk
//                ::= GV <object name> # Guard variable for one-time initialization
//                                     # No <type>

const char*
__demangle_tree::__parse_special_name(const char* first, const char* last)
{
    if (last - first > 2)
    {
        const char* t;
        switch (*first)
        {
        case 'T':
            switch (first[1])
            {
            case 'V':
                // TV <type>    # virtual table
                t = __parse_type(first+2, last);
                if (t != first+2 && __make<__vtable>(__root_))
                    first = t;
                break;
            case 'T':
                // TT <type>    # VTT structure (construction vtable index)
                t = __parse_type(first+2, last);
                if (t != first+2 && __make<__VTT>(__root_))
                    first = t;
                break;
            case 'I':
                // TI <type>    # typeinfo structure
                t = __parse_type(first+2, last);
                if (t != first+2 && __make<__typeinfo>(__root_))
                    first = t;
                break;
            case 'S':
                // TS <type>    # typeinfo name (null-terminated byte string)
                t = __parse_type(first+2, last);
                if (t != first+2 && __make<__typeinfo_name>(__root_))
                    first = t;
                break;
            case 'c':
                // Tc <call-offset> <call-offset> <base encoding>
                {
                const char* t0 = __parse_call_offset(first+2, last);
                if (t0 == first+2)
                    break;
                const char* t1 = __parse_call_offset(t0, last);
                if (t1 == t0)
                    break;
                t = __parse_encoding(t1, last);
                if (t != t1 && __make<__covariant_return_thunk>(__root_))
                    first = t;
                }
                break;
            default:
                // T <call-offset> <base encoding>
                {
                const char* t0 = __parse_call_offset(first+1, last);
                if (t0 == first+1)
                    break;
                t = __parse_encoding(t0, last);
                if (t != t0)
                {
                    if (first[2] == 'v')
                    {
                        if (__make<__virtual_thunk>(__root_))
                            first = t;
                    }
                    else
                    {
                        if (__make<__non_virtual_thunk>(__root_))
                            first = t;
                    }
                }
                }
                break;
            }
            break;
        case 'G':
            if (first[1] == 'V')
            {
                // GV <object name> # Guard variable for one-time initialization
                t = __parse_name(first+2, last);
                if (t != first+2 && __make<__guard_variable>(__root_))
                    first = t;
            }
            break;
        }
    }
    return first;
}

// <operator-name>
//                 ::= aa         # &&            
//                 ::= ad         # & (unary)
//                 ::= an         # &             
//                 ::= aN         # &=            
//                 ::= aS         # =             
//                 ::= at         # alignof (a type)
//                 ::= az         # alignof (an expression)
//                 ::= cl         # ()            
//                 ::= cm         # ,             
//                 ::= co         # ~             
//                 ::= cv <type>  # (cast)        
//                 ::= da         # delete[]
//                 ::= de         # * (unary)     
//                 ::= dl         # delete        
//                 ::= dv         # /             
//                 ::= dV         # /=            
//                 ::= eo         # ^             
//                 ::= eO         # ^=            
//                 ::= eq         # ==            
//                 ::= ge         # >=            
//                 ::= gt         # >             
//                 ::= ix         # []            
//                 ::= le         # <=            
//                 ::= ls         # <<            
//                 ::= lS         # <<=           
//                 ::= lt         # <             
//                 ::= mi         # -             
//                 ::= mI         # -=            
//                 ::= ml         # *             
//                 ::= mL         # *=            
//                 ::= mm         # -- (postfix in <expression> context)           
//                 ::= na         # new[]
//                 ::= ne         # !=            
//                 ::= ng         # - (unary)     
//                 ::= nt         # !             
//                 ::= nw         # new           
//                 ::= oo         # ||            
//                 ::= or         # |             
//                 ::= oR         # |=            
//                 ::= pm         # ->*           
//                 ::= pl         # +             
//                 ::= pL         # +=            
//                 ::= pp         # ++ (postfix in <expression> context)
//                 ::= ps         # + (unary)
//                 ::= pt         # ->            
//                 ::= qu         # ?             
//                 ::= rm         # %             
//                 ::= rM         # %=            
//                 ::= rs         # >>            
//                 ::= rS         # >>=           
//                 ::= st         # sizeof (a type)
//                 ::= sz         # sizeof (an expression)
//                 ::= v <digit> <source-name> # vendor extended operator

const char*
__demangle_tree::__parse_operator_name(const char* first, const char* last, int* type)
{
    if (last - first >= 2)
    {
        switch (*first)
        {
        case 'a':
            switch (first[1])
            {
            case 'a':
                // &&
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_logical_and>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_logical_and>())
                        first += 2;
                }
                break;
            case 'd':
                // & (unary)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_addressof>(__root_))
                        {
                            *type = 1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_addressof>())
                        first += 2;
                }
                break;
            case 'n':
                // &
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_bit_and>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_bit_and>())
                        first += 2;
                }
                break;
            case 'N':
                // &=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_and_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_and_equal>())
                        first += 2;
                }
                break;
            case 'S':
                // =
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_equal>())
                        first += 2;
                }
                break;
            case 't':
                // alignof (a type)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_alignof_type>(__root_))
                        {
                            *type = -1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_alignof_type>())
                        first += 2;
                }
                break;
            case 'z':
                // alignof (an expression)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_alignof_expression>(__root_))
                        {
                            *type = -1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_alignof_expression>())
                        first += 2;
                }
                break;
            }
            break;
        case 'c':
            switch (first[1])
            {
            case 'l':
                // ()
                if (__make<__operator_paren>())
                {
                    first += 2;
                    if (type)
                        *type = -1;
                }
                break;
            case 'm':
                // ,
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_comma>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_comma>())
                        first += 2;
                }
                break;
            case 'o':
                // ~
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_tilda>(__root_))
                        {
                            *type = 1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_tilda>())
                        first += 2;
                }
                break;
            case 'v':
                // cast <type>
                {
                const char* t = __parse_type(first+2, last, false);
                if (t != first+2)
                {
                    __node* cast_type = __root_;
                    if (type)
                    {
                        const char* t2 = __parse_expression(t, last);
                        if (t2 != t)
                        {
                            if (__make<__operator_cast>(cast_type, __root_))
                            {
                                *type = -1;
                                first = t2;
                            }
                        }
                    }
                    else
                    {
                        if (__make<__operator_cast>(cast_type))
                            first = t;
                    }
                }
                }
                break;
            }
            break;
        case 'd':
            switch (first[1])
            {
            case 'a':
                // delete[]
                if (__make<__operator_delete_array>())
                {
                    first += 2;
                    if (type)
                        *type = -1;
                }
                break;
            case 'e':
                // * (unary)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_dereference>(__root_))
                        {
                            *type = 1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_dereference>())
                        first += 2;
                }
                break;
            case 'l':
                // delete
                if (__make<__operator_delete>())
                {
                    first += 2;
                    if (type)
                        *type = -1;
                }
                break;
            case 'v':
                // /
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_divide>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_divide>())
                        first += 2;
                }
                break;
            case 'V':
                // /=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_divide_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_divide_equal>())
                        first += 2;
                }
                break;
            }
            break;
        case 'e':
            switch (first[1])
            {
            case 'o':
                // ^
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_xor>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_xor>())
                        first += 2;
                }
                break;
            case 'O':
                // ^=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_xor_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_xor_equal>())
                        first += 2;
                }
                break;
            case 'q':
                // ==
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_equality>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_equality>())
                        first += 2;
                }
                break;
            }
            break;
        case 'g':
            switch (first[1])
            {
            case 'e':
                // >=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_greater_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_greater_equal>())
                        first += 2;
                }
                break;
            case 't':
                // >
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_greater>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_greater>())
                        first += 2;
                }
                break;
            }
            break;
        case 'i':
            // []
            if (first[1] == 'x' && __make<__operator_brackets>())
                {
                first += 2;
                    if (type)
                        *type = -1;
                }
            break;
        case 'l':
            switch (first[1])
            {
            case 'e':
                // <=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_less_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_less_equal>())
                        first += 2;
                }
                break;
            case 's':
                // <<
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_left_shift>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_left_shift>())
                        first += 2;
                }
                break;
            case 'S':
                // <<=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_left_shift_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_left_shift_equal>())
                        first += 2;
                }
                break;
            case 't':
                // <
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_less>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_less>())
                        first += 2;
                }
                break;
            }
            break;
        case 'm':
            switch (first[1])
            {
            case 'i':
                // -
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_minus>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_minus>())
                        first += 2;
                }
                break;
            case 'I':
                // -=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_minus_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_minus_equal>())
                        first += 2;
                }
                break;
            case 'l':
                // *
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_times>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_times>())
                        first += 2;
                }
                break;
            case 'L':
                // *=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_times_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_times_equal>())
                        first += 2;
                }
                break;
            case 'm':
                // -- (postfix in <expression> context)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_decrement>(false, __root_))
                        {
                            *type = 1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_decrement>())
                        first += 2;
                }
                break;
            }
            break;
        case 'n':
            switch (first[1])
            {
            case 'a':
                // new[]
                if (__make<__operator_new_array>())
                {
                    first += 2;
                    if (type)
                        *type = -1;
                }
                break;
            case 'e':
                // !=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_not_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_not_equal>())
                        first += 2;
                }
                break;
            case 'g':
                // - (unary)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_negate>(__root_))
                        {
                            *type = 1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_negate>())
                        first += 2;
                }
                break;
            case 't':
                // !
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_logical_not>(__root_))
                        {
                            *type = 1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_logical_not>())
                        first += 2;
                }
                break;
            case 'w':
                // new
                if (__make<__operator_new>())
                {
                    first += 2;
                    if (type)
                        *type = -1;
                }
                break;
            }
            break;
        case 'o':
            switch (first[1])
            {
            case 'o':
                // ||
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_logical_or>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_logical_or>())
                        first += 2;
                }
                break;
            case 'r':
                // |
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_bit_or>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_bit_or>())
                        first += 2;
                }
                break;
            case 'R':
                // |=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_or_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_or_equal>())
                        first += 2;
                }
                break;
            }
            break;
        case 'p':
            switch (first[1])
            {
            case 'm':
                // ->*
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_pointer_to_member>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_pointer_to_member>())
                        first += 2;
                }
                break;
            case 'l':
                // +
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_plus>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_plus>())
                        first += 2;
                }
                break;
            case 'L':
                // +=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_plus_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_plus_equal>())
                        first += 2;
                }
                break;
            case 'p':
                // ++ (postfix in <expression> context)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_increment>(false, __root_))
                        {
                            *type = 1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_increment>())
                        first += 2;
                }
                break;
            case 's':
                // + (unary)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_unary_plus>(__root_))
                        {
                            *type = 1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_unary_plus>())
                        first += 2;
                }
                break;
            case 't':
                // ->
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_arrow>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_arrow>())
                        first += 2;
                }
                break;
            }
            break;
        case 'q':
            // ?
            if (first[1] == 'u')
            {
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            __node* op2 = __root_;
                            const char* t3 = __parse_expression(t2, last);
                            if (t3 != t2)
                            {
                                if (__make<__operator_conditional>(op1, op2, __root_))
                                {
                                    *type = 3;
                                    first = t3;
                                }
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_conditional>())
                        first += 2;
                }
            }
            break;
        case 'r':
            switch (first[1])
            {
            case 'm':
                // %
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_mod>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_mod>())
                        first += 2;
                }
                break;
            case 'M':
                // %=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_mod_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_mod_equal>())
                        first += 2;
                }
                break;
            case 's':
                // >>
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_right_shift>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_right_shift>())
                        first += 2;
                }
                break;
            case 'S':
                // >>=
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        __node* op1 = __root_;
                        const char* t2 = __parse_expression(t, last);
                        if (t != t2)
                        {
                            if (__make<__operator_right_shift_equal>(op1, __root_))
                            {
                                *type = 2;
                                first = t2;
                            }
                        }
                    }
                }
                else
                {
                    if (__make<__operator_right_shift_equal>())
                        first += 2;
                }
                break;
            }
            break;
        case 's':
            switch (first[1])
            {
            case 't':
                // sizeof (a type)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_sizeof_type>(__root_))
                        {
                            *type = -1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_sizeof_type>())
                        first += 2;
                }
                break;
            case 'z':
                // sizeof (an expression)
                if (type)
                {
                    const char* t = __parse_expression(first+2, last);
                    if (t != first+2)
                    {
                        if (__make<__operator_sizeof_expression>(__root_))
                        {
                            *type = -1;
                            first = t;
                        }
                    }
                }
                else
                {
                    if (__make<__operator_sizeof_expression>())
                        first += 2;
                }
                break;
            }
            break;
        }
    }
    return first;
}

// <source-name> ::= <positive length number> <identifier>

const char*
__demangle_tree::__parse_source_name(const char* first, const char* last)
{
    if (first != last)
    {
        char c = *first;
        if ('1' <= c && c <= '9' && first+1 != last)
        {
            const char* t = first+1;
            size_t n = c - '0';
            for (c = *t; '0' <= c && c <= '9'; c = *t)
            {
                n = n * 10 + c - '0';
                if (++t == last)
                    return first;
            }
            if (last - t >= n && __make<__source_name>(t, n))
                first = t + n;
        }
    }
    return first;
}

// <unqualified-name> ::= <operator-name>
//                    ::= <ctor-dtor-name>
//                    ::= <source-name>   
//                    ::= <unnamed-type-name>

const char*
__demangle_tree::__parse_unqualified_name(const char* first, const char* last)
{
    const char* t = __parse_source_name(first, last);
    if (t == first)
    {
        t = __parse_ctor_dtor_name(first, last);
        if (t == first)
        {
            t = __parse_operator_name(first, last);
            if (t == first)
                first = __parse_unnamed_type_name(first, last);
            else
                first = t;
        }
        else
            first = t;
    }
    else
        first = t;
    return first;
}

// <unscoped-name> ::= <unqualified-name>
//                 ::= St <unqualified-name>   # ::std::
// extension       ::= StL<unqualified-name>

const char*
__demangle_tree::__parse_unscoped_name(const char* first, const char* last)
{
    if (last - first >= 2)
    {
        const char* t0 = first;
        if (first[0] == 'S' && first[1] == 't')
        {
            t0 += 2;
            if (t0 != last && *t0 == 'L')
                ++t0;
        }
        const char* t1 = __parse_unqualified_name(t0, last);
        if (t1 != t0)
        {
            if (t0 != first)
            {
                __node* name = __root_;
                if (__make<__std_qualified_name>())
                {
                    if (__make<__nested_delimeter>(__root_, name))
                        first = t1;
                }
            }
            else
                first = t1;
        }
    }
    return first;
}

// <nested-name> ::= N [<CV-qualifiers>] <prefix> <unqualified-name> E
//               ::= N [<CV-qualifiers>] <template-prefix> <template-args> E
// 
// <prefix> ::= <prefix> <unqualified-name>
//          ::= <template-prefix> <template-args>
//          ::= <template-param>
//          ::= <decltype>
//          ::= # empty
//          ::= <substitution>
//          ::= <prefix> <data-member-prefix>
//  extension ::= L
// 
// <template-prefix> ::= <prefix> <template unqualified-name>
//                   ::= <template-param>
//                   ::= <substitution>

const char*
__demangle_tree::__parse_nested_name(const char* first, const char* last)
{
    if (first != last && *first == 'N')
    {
        unsigned cv = 0;
        const char* t0 = __parse_cv_qualifiers(first+1, last, cv, true);
        __node* prev = NULL;
        if (last - t0 >= 2 && t0[0] == 'S' && t0[1] == 't')
        {
            t0 += 2;
            if (!__make<__std_qualified_name>())
                return first;
            prev = __root_;
        }
        while (t0 != last)
        {
            bool can_sub = true;
            bool make_nested = true;
            const char* t1;
            switch (*t0)
            {
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                t1 = __parse_source_name(t0, last);
                if (t1 == t0 || t1 == last)
                    return first;
                if (*t1 == 'M')
                {
                    // This is a data-member-prefix
                    ++t1;
                }
                else if (*t1 == 'I')
                {
                    // has following <template-args>
                    if (prev)
                    {
                        if (!__make<__nested_delimeter>(prev, __root_))
                            return first;
                        make_nested = false;
                    }
                    if (__sub_end_ == __sub_cap_)
                    {
                        __status_ = memory_alloc_failure;
                        return first;
                    }
                    else
                        *__sub_end_++ = __root_;
                    const char* t2 = __parse_template_args(t1, last);
                    if (t2 == t1)
                        return first;
                    t1 = t2;
                }
                break;
            case 'D':
                if (t0+1 != last && (t0[1] == 't' || t0[1] == 'T'))
                {
                    t1 = __parse_decltype(t0, last);
                    break;
                }
                // check for Dt, DT here, else drop through
            case 'C':
                t1 = __parse_ctor_dtor_name(t0, last);
                if (t1 == t0 || t1 == last)
                    return first;
                if (*t1 == 'I')
                {
                    // has following <template-args>
                    if (prev)
                    {
                        if (!__make<__nested_delimeter>(prev, __root_))
                            return first;
                        make_nested = false;
                    }
                    if (__sub_end_ == __sub_cap_)
                    {
                        __status_ = memory_alloc_failure;
                        return first;
                    }
                    else
                        *__sub_end_++ = __root_;
                    const char* t2 = __parse_template_args(t1, last);
                    if (t2 == t1)
                        return first;
                    t1 = t2;
                }
                break;
            case 'U':
                assert(!"__parse_nested_name U");
                // could have following <template-args>
                break;
            case 'T':
                t1 = __parse_template_param(t0, last);
                if (t1 == t0 || t1 == last)
                    return first;
                if (*t1 == 'I')
                {
                    // has following <template-args>
                    if (prev)
                    {
                        if (!__make<__nested_delimeter>(prev, __root_))
                            return first;
                        make_nested = false;
                    }
                    if (__sub_end_ == __sub_cap_)
                    {
                        __status_ = memory_alloc_failure;
                        return first;
                    }
                    else
                        *__sub_end_++ = __root_;
                    const char* t2 = __parse_template_args(t1, last);
                    if (t2 == t1)
                        return first;
                    t1 = t2;
                }
                break;
            case 'S':
                t1 = __parse_substitution(t0, last);
               if (t1 == t0 || t1 == last)
                    return first;
                if (*t1 == 'I')
                {
                    const char* t2 = __parse_template_args(t1, last);
                    if (t2 == t1)
                        return first;
                    t1 = t2;
                }
                else
                    can_sub = false;
                break;
            case 'L':
                // extension: ignore L here
                ++t0;
                continue;
            default:
                t1 = __parse_operator_name(t0, last);
                if (t1 == t0 || t1 == last)
                    return first;
                if (*t1 == 'I')
                {
                    // has following <template-args>
                    if (prev)
                    {
                        if (!__make<__nested_delimeter>(prev, __root_))
                            return first;
                        make_nested = false;
                    }
                    if (__sub_end_ == __sub_cap_)
                    {
                        __status_ = memory_alloc_failure;
                        return first;
                    }
                    else
                        *__sub_end_++ = __root_;
                    const char* t2 = __parse_template_args(t1, last);
                    if (t2 == t1)
                        return first;
                    t1 = t2;
                }
                break;
            }
            if (t1 == t0 || t1 == last)
                return first;
            if (prev && make_nested)
            {
                if (!__make<__nested_delimeter>(prev, __root_))
                    return first;
                can_sub = true;
            }
            if (can_sub && *t1 != 'E')
            {
                if (__sub_end_ == __sub_cap_)
                {
                    __status_ = memory_alloc_failure;
                    return first;
                }
                else
                    *__sub_end_++ = __root_;
            }
            if (*t1 == 'E')
            {
                if (cv != 0)
                {
                    if (!__make<__cv_qualifiers>(cv, __root_))
                        return first;
                }
                first = t1+1;
                break;
            }
            prev = __root_;
            t0 = t1;
        }
    }
    return first;
}

// <template-arg> ::= <type>                                             # type or template
//                ::= X <expression> E                                   # expression
//                ::= <expr-primary>                                     # simple expressions
//                ::= J <template-arg>* E                                # argument pack
//                ::= LZ <encoding> E                                    # extension

const char*
__demangle_tree::__parse_template_arg(const char* first, const char* last)
{
    if (first != last)
    {
        const char* t;
        switch (*first)
        {
        case 'X':
            t = __parse_expression(first+1, last);
            if (t != first+1)
            {
                if (t != last && *t == 'E')
                    first = t+1;
            }
            break;
        case 'J':
            t = first+1;
            if (t == last)
                return first;
            if (*t == 'E')
            {
                if (__make<__list>((__node*)0))
                    first = t+1;
            }
            else
            {
                __node* list = NULL;
                __node* prev = NULL;
                do
                {
                    const char* t2 = __parse_template_arg(t, last);
                    if (t2 == t || !__make<__list>(__root_))
                        return first;
                    if (list == 0)
                        list = __root_;
                    if (prev)
                    {
                        prev->__right_ = __root_;
                        __root_->__size_ = prev->__size_ + 1;
                    }
                    prev = __root_;
                    t = t2;
                } while (t != last && *t != 'E');
                first = t+1;
                __root_ = list;
            }
            break;
        case 'L':
            // <expr-primary> or LZ <encoding> E
            if (first+1 != last && first[1] == 'Z')
            {
                t = __parse_encoding(first+2, last);
                if (t != first+2 && t != last && *t == 'E')
                    first = t+1;
            }
            else
                first = __parse_expr_primary(first, last);
            break;
        default:
            // <type>
            first = __parse_type(first, last);
            break;
        }
    }
    return first;
}

// <template-args> ::= I <template-arg>* E
//     extension, the abi says <template-arg>+

const char*
__demangle_tree::__parse_template_args(const char* first, const char* last)
{
    if (last - first >= 2 && *first == 'I')
    {
        __node* args = NULL;
        __node* prev = NULL;
        __node* name = __root_;
        bool prev_tag_templates = __tag_templates_;
        __tag_templates_ = false;
        if (prev_tag_templates)
            __t_end_ = __t_begin_;
        const char* t = first+1;
        while (*t != 'E')
        {
            const char* t2 = __parse_template_arg(t, last);
            if (t2 == t || t2 == last)
                break;
            if (!__make<__list>(__root_))
                return first;
            if (args == 0)
                args = __root_;
            if (prev)
            {
                prev->__right_ = __root_;
                __root_->__size_ = prev->__size_ + 1;
            }
            prev = __root_;
            if (prev_tag_templates)
            {
                if (__t_end_ == __t_cap_)
                {
                    __status_ = memory_alloc_failure;
                    return first;
                }
                if (__root_->__left_)
                    *__t_end_++ = __root_->__left_;
                else
                    *__t_end_++ = __root_;
            }
            t = t2;
        }
        if (t != last && *t == 'E')
        {
            if (__make<__template_args>(name, args))
                first = t+1;
        }
        __tag_templates_ = prev_tag_templates;
    }
    return first;
}

// <substitution> ::= S <seq-id> _
//                ::= S_
// <substitution> ::= Sa # ::std::allocator
// <substitution> ::= Sb # ::std::basic_string
// <substitution> ::= Ss # ::std::basic_string < char,
//                                               ::std::char_traits<char>,
//                                               ::std::allocator<char> >
// <substitution> ::= Si # ::std::basic_istream<char,  std::char_traits<char> >
// <substitution> ::= So # ::std::basic_ostream<char,  std::char_traits<char> >
// <substitution> ::= Sd # ::std::basic_iostream<char, std::char_traits<char> >

const char*
__demangle_tree::__parse_substitution(const char* first, const char* last)
{
    if (last - first >= 2)
    {
        if (*first == 'S')
        {
            switch (first[1])
            {
            case 'a':
                if (__make<__sub_allocator>())
                     first += 2;
                break;
            case 'b':
                if (__make<__sub_basic_string>())
                     first += 2;
                break;
            case 's':
                if (__make<__sub_string>())
                     first += 2;
                break;
            case 'i':
                if (__make<__sub_istream>())
                     first += 2;
                break;
            case 'o':
                if (__make<__sub_ostream>())
                     first += 2;
                break;
            case 'd':
                if (__make<__sub_iostream>())
                     first += 2;
                break;
            case '_':
                if (__sub_begin_ != __sub_end_)
                {
                    if (__make<__sub>(*__sub_begin_))
                        first += 2;
                }
                break;
            default:
                if (isdigit(first[1]) || isupper(first[1]))
                {
                    size_t sub = 0;
                    const char* t = first+1;
                    if (isdigit(*t))
                        sub = *t - '0';
                    else
                        sub = *t - 'A' + 10;
                    for (++t; t != last && (isdigit(*t) || isupper(*t)); ++t)
                    {
                        sub *= 36;
                        if (isdigit(*t))
                            sub += *t - '0';
                        else
                            sub += *t - 'A' + 10;
                    }
                    if (t == last || *t != '_')
                        return first;
                    ++sub;
                    if (sub < __sub_end_ - __sub_begin_)
                    {
                        if (__make<__sub>(__sub_begin_[sub]))
                            first = t+1;
                    }
                }
                break;
            }
        }
    }
    return first;
}

// <name> ::= <nested-name>
//        ::= <local-name> # See Scope Encoding below
//        ::= <unscoped-template-name> <template-args>
//        ::= <unscoped-name>

const char*
__demangle_tree::__parse_name(const char* first, const char* last)
{
    if (first != last)
    {
        const char* t0 = first;
        // extension: ignore L here
        if (*t0 == 'L')
            ++t0;
        const char* t = __parse_nested_name(t0, last);
        if (t == t0)
        {
            t = __parse_local_name(t0, last);
            if (t == t0)
            {
                // not <nested-name> nor <local-name>
                // Try to parse <unscoped-template-name> <template-args> or
                //   <unscoped-name> which are nearly ambiguous.
                //   This logic occurs nowhere else.
                if (last - t0 >= 2)
                {
                    if (t0[0] == 'S' && (t0[1] == '_'   ||
                                            isdigit(t0[1]) ||
                                            isupper(t0[1]) ||
                                            t0[1] == 'a'   ||
                                            t0[1] == 'b'))
                    {
                        t = __parse_substitution(t0, last);
                        if (t != t0)
                        {
                            const char* t2 = __parse_template_args(t, last);
                            if (t2 != t)
                                first = t2;
                        }
                    }
                    else  // Not a substitution, except maybe St
                    {
                        t = __parse_unscoped_name(t0, last);
                        if (t != t0)
                        {
                            // unscoped-name might be <unscoped-template-name>
                            if (t != last && *t == 'I')
                            {
                                if (__sub_end_ == __sub_cap_)
                                {
                                    __status_ = memory_alloc_failure;
                                    return first;
                                }
                                *__sub_end_++ = __root_;
                                const char* t2 = __parse_template_args(t, last);
                                if (t2 != t)
                                    first = t2;
                            }
                            else
                            {
                                // <unscoped-name>
                                first = t;
                            }
                        }
                    }
                }
            }
            else
                first = t;
        }
        else
            first = t;
    }
    return first;
}

// extension
// <dot-suffix> := .<anything and everything>

const char*
__demangle_tree::__parse_dot_suffix(const char* first, const char* last)
{
    if (first != last && *first == '.')
    {
        if (__make<__dot_suffix>(__root_, first, last-first))
            first = last;
    }
    return first;
}

// <encoding> ::= <function name> <bare-function-type>
//            ::= <data name>
//            ::= <special-name>

const char*
__demangle_tree::__parse_encoding(const char* first, const char* last)
{
    const char* t = __parse_name(first, last);
    if (t != first)
    {
        if (t != last && *t != 'E' && *t != '.')
        {
            __node* name = __root_;
            bool has_return = name->ends_with_template() &&
                             !name->is_ctor_dtor_conv();
            __node* ret = NULL;
            const char* t2;
            __tag_templates_ = false;
            if (has_return)
            {
                t2 = __parse_type(t, last);
                if (t2 != t)
                {
                    ret = __root_;
                    t = t2;
                }
                else
                    return first;
            }
            t2 = __parse_bare_function_type(t, last);
            if (t2 != t)
            {
                if (dynamic_cast<__void*>(__root_->__left_) != NULL)
                    __root_->__left_ = NULL;
                if (__make<__function_signature>(ret, __root_))
                {
                    __node* cv = name->extract_cv(name);
                    if (__make<__function>(name, __root_))
                    {
                        if (cv)
                        {
                            cv->__left_ = __root_;
                            cv->__size_ <<= 5;
                            __root_ = cv;
                        }
                        first = t2;
                    }
                }
            }
            __tag_templates_ = true;
        }
        else
            first = t;
    }
    else
        first = __parse_special_name(first, last);
    return first;
}

// <mangled-name> ::= _Z<encoding>
//                ::= <type>

void
__demangle_tree::__parse()
{
    if (__mangled_name_begin_ == __mangled_name_end_)
    {
        __status_ = invalid_mangled_name;
        return;
    }
    const char* t = NULL;
    if (__mangled_name_end_ - __mangled_name_begin_ >= 2 &&
                         __mangled_name_begin_[0] == '_' &&
                         __mangled_name_begin_[1] == 'Z')
    {
        t = __parse_encoding(__mangled_name_begin_+2, __mangled_name_end_);
        if (t != __mangled_name_begin_+2 && t != __mangled_name_end_ && *t == '.')
            t = __parse_dot_suffix(t, __mangled_name_end_);
    }
    else
        t = __parse_type(__mangled_name_begin_, __mangled_name_end_);
    if (t == __mangled_name_end_ && __root_)
    {
        if (__fix_forward_references_)
        {
            if (__root_->fix_forward_references(__t_begin_, __t_end_))
               __status_ = success;
        }
        else
           __status_ = success;
    }
}

#pragma GCC visibility pop
#pragma GCC visibility push(default)

__demangle_tree
__demangle(const char* mangled_name, char* buf, size_t bs)
{
    __demangle_tree t(mangled_name, buf, bs);
    if (t.__status() == invalid_mangled_name)
        t.__parse();
    return t;
}

__demangle_tree
__demangle(const char* mangled_name)
{
    return __demangle(mangled_name, 0, 0);
}

char*
__demangle(__demangle_tree dmg_tree, char* buf, size_t* n, int* status)
{
    if (dmg_tree.__status() != success)
    {
        if (status)
            *status = dmg_tree.__status();
        return NULL;
    }
#ifdef DEBUGGING
display(dmg_tree.__root_);
printf("\n");
#endif
    const size_t bs = buf == NULL ? 0 : *n;
    ptrdiff_t sm = dmg_tree.__mangled_name_end_ - dmg_tree.__mangled_name_begin_;
    ptrdiff_t est = sm + 50 * (dmg_tree.__node_end_ - dmg_tree.__node_begin_);
    const unsigned N = 4096;
    char tmp[N];
    ptrdiff_t s;
    if (est <= bs)
    {
        char* e = dmg_tree.__get_demangled_name(buf);
        *e++ = '\0';
        s = e - buf;
    }
    else if (est <= N)
    {
        char* e = dmg_tree.__get_demangled_name(tmp);
        *e++ = '\0';
        s = e - tmp;
    }
    else
        s = dmg_tree.size() + 1;
    if (s > bs)
    {
        buf = static_cast<char*>(realloc(buf, s));
        if (buf == NULL)
        {
            if (status)
                *status = memory_alloc_failure;
            return NULL;
        }
        if (n)
            *n = s;
    }
    if (est > bs)
    {
        if (est <= N)
            strncpy(buf, tmp, s);
        else
            *dmg_tree.__get_demangled_name(buf) = '\0';
    }
    if (status)
        *status = success;
    return buf;
}

}  // __libcxxabi

extern "C"
{

char*
__cxa_demangle(const char* mangled_name, char* buf, size_t* n, int* status)
{
    if (mangled_name == NULL || (buf != NULL && n == NULL))
    {
        if (status)
            *status = __libcxxabi::invalid_args;
        return NULL;
    }
    const size_t bs = 64 * 1024;
    char static_buf[bs];

    buf = __libcxxabi::__demangle(__libcxxabi::__demangle(mangled_name,
                                                          static_buf, bs),
                                  buf, n, status);
    return buf;
}

}  // extern "C"

}  // abi
