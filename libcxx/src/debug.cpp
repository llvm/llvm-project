//===-------------------------- debug.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define _LIBCPP_DEBUG2 1
#include "__config"
#include "__debug"
#include "functional"
#include "algorithm"
#include "__hash_table"
#include "mutex"

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_VISIBLE
__libcpp_db*
__get_db()
{
    static __libcpp_db db;
    return &db;
};

_LIBCPP_VISIBLE
const __libcpp_db*
__get_const_db()
{
    return __get_db();
}

namespace
{

typedef mutex mutex_type;
typedef lock_guard<mutex_type> WLock;
typedef lock_guard<mutex_type> RLock;

mutex_type&
mut()
{
    static mutex_type m;
    return m;
}

}  // unnamed namespace

__i_node::~__i_node()
{
    if (__next_)
    {
        __next_->~__i_node();
        free(__next_);
    }
}

__c_node::~__c_node()
{
    free(beg_);
    if (__next_)
    {
        __next_->~__c_node();
        free(__next_);
    }
}

__libcpp_db::__libcpp_db()
    : __cbeg_(nullptr),
      __cend_(nullptr),
      __csz_(0),
      __ibeg_(nullptr),
      __iend_(nullptr),
      __isz_(0)
{
}

__libcpp_db::~__libcpp_db()
{
    if (__cbeg_)
    {
        for (__c_node** p = __cbeg_; p != __cend_; ++p)
        {
            if (*p != nullptr)
            {
                (*p)->~__c_node();
                free(*p);
            }
        }
        free(__cbeg_);
    }
    if (__ibeg_)
    {
        for (__i_node** p = __ibeg_; p != __iend_; ++p)
        {
            if (*p != nullptr)
            {
                (*p)->~__i_node();
                free(*p);
            }
        }
        free(__ibeg_);
    }
}

void*
__libcpp_db::__find_c_from_i(void* __i) const
{
    RLock _(mut());
    __i_node* i = __find_iterator(__i);
    _LIBCPP_ASSERT(i != nullptr, "iterator constructed in translation unit with debug mode not enabled."
                   "  #define _LIBCPP_DEBUG2 1 for that translation unit.");
    return i->__c_ != nullptr ? i->__c_->__c_ : nullptr;
}

void
__libcpp_db::__insert_ic(void* __i, const void* __c)
{
    WLock _(mut());
    __i_node* i = __insert_iterator(__i);
    const char* errmsg =
        "Container constructed in a translation unit with debug mode disabled."
        " But it is being used in a translation unit with debug mode enabled."
        " Enable it in the other translation unit with #define _LIBCPP_DEBUG2 1";
    _LIBCPP_ASSERT(__cbeg_ != __cend_, errmsg);
    size_t hc = hash<const void*>()(__c) % (__cend_ - __cbeg_);
    __c_node* c = __cbeg_[hc];
    _LIBCPP_ASSERT(c != nullptr, errmsg);
    while (c->__c_ != __c)
    {
        c = c->__next_;
        _LIBCPP_ASSERT(c != nullptr, errmsg);
    }
    c->__add(i);
    i->__c_ = c;
}

__c_node*
__libcpp_db::__insert_c(void* __c)
{
    WLock _(mut());
    if (__csz_ + 1 > __cend_ - __cbeg_)
    {
        size_t nc = __next_prime(2*(__cend_ - __cbeg_) + 1);
        __c_node** cbeg = (__c_node**)calloc(nc, sizeof(void*));
        if (cbeg == nullptr)
            throw bad_alloc();
        for (__c_node** p = __cbeg_; p != __cend_; ++p)
        {
            __c_node* q = *p;
            while (q != nullptr)
            {
                size_t h = hash<void*>()(q->__c_) % nc;
                __c_node* r = q->__next_;
                q->__next_ = cbeg[h];
                cbeg[h] = q;
                q = r;
            }
        }
        free(__cbeg_);
        __cbeg_ = cbeg;
        __cend_ = __cbeg_ + nc;
    }
    size_t hc = hash<void*>()(__c) % (__cend_ - __cbeg_);
    __c_node* p = __cbeg_[hc];
    __c_node* r = __cbeg_[hc] = (__c_node*)malloc(sizeof(__c_node));
    if (__cbeg_[hc] == nullptr)
        throw bad_alloc();
    r->__c_ = __c;
    r->__next_ = p;
    ++__csz_;
    return r;
}

void
__libcpp_db::__erase_i(void* __i)
{
    WLock _(mut());
    if (__ibeg_ != __iend_)
    {
        size_t hi = hash<void*>()(__i) % (__iend_ - __ibeg_);
        __i_node* p = __ibeg_[hi];
        if (p != nullptr)
        {
            __i_node* q = nullptr;
            while (p->__i_ != __i)
            {
                q = p;
                p = p->__next_;
                if (p == nullptr)
                    return;
            }
            if (q == nullptr)
                __ibeg_[hi] = p->__next_;
            else
                q->__next_ = p->__next_;
            __c_node* c = p->__c_;
            free(p);
            --__isz_;
            if (c != nullptr)
                c->__remove(p);
        }
    }
}

void
__libcpp_db::__invalidate_all(void* __c)
{
    WLock _(mut());
    size_t hc = hash<void*>()(__c) % (__cend_ - __cbeg_);
    __c_node* p = __cbeg_[hc];
    _LIBCPP_ASSERT(p != nullptr, "debug mode internal logic error __invalidate_all A");
    while (p->__c_ != __c)
    {
        p = p->__next_;
        _LIBCPP_ASSERT(p != nullptr, "debug mode internal logic error __invalidate_all B");
    }
    while (p->end_ != p->beg_)
    {
        --p->end_;
        (*p->end_)->__c_ = nullptr;
    }
}

__c_node*
__libcpp_db::__find_c_and_lock(void* __c) const
{
    mut().lock();
    size_t hc = hash<void*>()(__c) % (__cend_ - __cbeg_);
    __c_node* p = __cbeg_[hc];
    _LIBCPP_ASSERT(p != nullptr, "debug mode internal logic error __find_c_and_lock A");
    while (p->__c_ != __c)
    {
        p = p->__next_;
        _LIBCPP_ASSERT(p != nullptr, "debug mode internal logic error __find_c_and_lock B");
    }
    return p;
}

__c_node*
__libcpp_db::__find_c(void* __c) const
{
    size_t hc = hash<void*>()(__c) % (__cend_ - __cbeg_);
    __c_node* p = __cbeg_[hc];
    _LIBCPP_ASSERT(p != nullptr, "debug mode internal logic error __find_c A");
    while (p->__c_ != __c)
    {
        p = p->__next_;
        _LIBCPP_ASSERT(p != nullptr, "debug mode internal logic error __find_c B");
    }
    return p;
}

void
__libcpp_db::unlock() const
{
    mut().unlock();
}

void
__libcpp_db::__erase_c(void* __c)
{
    WLock _(mut());
    size_t hc = hash<void*>()(__c) % (__cend_ - __cbeg_);
    __c_node* p = __cbeg_[hc];
    __c_node* q = nullptr;
    _LIBCPP_ASSERT(p != nullptr, "debug mode internal logic error __erase_c A");
    while (p->__c_ != __c)
    {
        q = p;
        p = p->__next_;
        _LIBCPP_ASSERT(p != nullptr, "debug mode internal logic error __erase_c B");
    }
    if (q == nullptr)
        __cbeg_[hc] = p->__next_;
    else
        q->__next_ = p->__next_;
    while (p->end_ != p->beg_)
    {
        --p->end_;
        (*p->end_)->__c_ = nullptr;
    }
    free(p->beg_);
    free(p);
    --__csz_;
}

void
__libcpp_db::__iterator_copy(void* __i, const void* __i0)
{
    WLock _(mut());
    __i_node* i = __find_iterator(__i);
    __i_node* i0 = __find_iterator(__i0);
    __c_node* c0 = i0 != nullptr ? i0->__c_ : nullptr;
    if (i == nullptr && c0 != nullptr)
        i = __insert_iterator(__i);
    __c_node* c = i != nullptr ? i->__c_ : nullptr;
    if (c != c0)
    {
        if (c != nullptr)
            c->__remove(i);
        if (i != nullptr)
        {
            i->__c_ = nullptr;
            if (c0 != nullptr)
            {
                i->__c_ = c0;
                i->__c_->__add(i);
            }
        }
    }
}

bool
__libcpp_db::__dereferenceable(const void* __i) const
{
    RLock _(mut());
    __i_node* i = __find_iterator(__i);
    return i != nullptr && i->__c_ != nullptr && i->__c_->__dereferenceable(__i);
}

bool
__libcpp_db::__decrementable(const void* __i) const
{
    RLock _(mut());
    __i_node* i = __find_iterator(__i);
    return i != nullptr && i->__c_ != nullptr && i->__c_->__decrementable(__i);
}

bool
__libcpp_db::__addable(const void* __i, ptrdiff_t __n) const
{
    RLock _(mut());
    __i_node* i = __find_iterator(__i);
    return i != nullptr && i->__c_ != nullptr && i->__c_->__addable(__i, __n);
}

bool
__libcpp_db::__subscriptable(const void* __i, ptrdiff_t __n) const
{
    RLock _(mut());
    __i_node* i = __find_iterator(__i);
    return i != nullptr && i->__c_ != nullptr && i->__c_->__subscriptable(__i, __n);
}

bool
__libcpp_db::__comparable(const void* __i, const void* __j) const
{
    RLock _(mut());
    __i_node* i = __find_iterator(__i);
    __i_node* j = __find_iterator(__j);
    __c_node* ci = i != nullptr ? i->__c_ : nullptr;
    __c_node* cj = j != nullptr ? j->__c_ : nullptr;
    return ci != nullptr && ci == cj;
}

void
__libcpp_db::swap(void* c1, void* c2)
{
    WLock _(mut());
    size_t hc = hash<void*>()(c1) % (__cend_ - __cbeg_);
    __c_node* p1 = __cbeg_[hc];
    _LIBCPP_ASSERT(p1 != nullptr, "debug mode internal logic error swap A");
    while (p1->__c_ != c1)
    {
        p1 = p1->__next_;
        _LIBCPP_ASSERT(p1 != nullptr, "debug mode internal logic error swap B");
    }
    hc = hash<void*>()(c2) % (__cend_ - __cbeg_);
    __c_node* p2 = __cbeg_[hc];
    _LIBCPP_ASSERT(p2 != nullptr, "debug mode internal logic error swap C");
    while (p2->__c_ != c2)
    {
        p2 = p2->__next_;
        _LIBCPP_ASSERT(p2 != nullptr, "debug mode internal logic error swap D");
    }
    std::swap(p1->beg_, p2->beg_);
    std::swap(p1->end_, p2->end_);
    std::swap(p1->cap_, p2->cap_);
    for (__i_node** p = p1->beg_; p != p1->end_; ++p)
        (*p)->__c_ = p1;
    for (__i_node** p = p2->beg_; p != p2->end_; ++p)
        (*p)->__c_ = p2;
}

void
__libcpp_db::__insert_i(void* __i)
{
    WLock _(mut());
    __insert_iterator(__i);
}

void
__c_node::__add(__i_node* i)
{
    if (end_ == cap_)
    {
        size_t nc = 2*(cap_ - beg_);
        if (nc == 0)
            nc = 1;
        __i_node** beg = (__i_node**)malloc(nc * sizeof(__i_node*));
        if (beg == nullptr)
            throw bad_alloc();
        if (nc > 1)
            memcpy(beg, beg_, nc/2*sizeof(__i_node*));
        free(beg_);
        beg_ = beg;
        end_ = beg_ + nc/2;
        cap_ = beg_ + nc;
    }
    *end_++ = i;
}

// private api

_LIBCPP_HIDDEN
__i_node*
__libcpp_db::__insert_iterator(void* __i)
{
    if (__isz_ + 1 > __iend_ - __ibeg_)
    {
        size_t nc = __next_prime(2*(__iend_ - __ibeg_) + 1);
        __i_node** ibeg = (__i_node**)calloc(nc, sizeof(void*));
        if (ibeg == nullptr)
            throw bad_alloc();
        for (__i_node** p = __ibeg_; p != __iend_; ++p)
        {
            __i_node* q = *p;
            while (q != nullptr)
            {
                size_t h = hash<void*>()(q->__i_) % nc;
                __i_node* r = q->__next_;
                q->__next_ = ibeg[h];
                ibeg[h] = q;
                q = r;
            }
        }
        free(__ibeg_);
        __ibeg_ = ibeg;
        __iend_ = __ibeg_ + nc;
    }
    size_t hi = hash<void*>()(__i) % (__iend_ - __ibeg_);
    __i_node* p = __ibeg_[hi];
    __i_node* r = __ibeg_[hi] = (__i_node*)malloc(sizeof(__i_node));
    if (r == nullptr)
        throw bad_alloc();
    ::new(r) __i_node(__i, p, nullptr);
    ++__isz_;
    return r;
}

_LIBCPP_HIDDEN
__i_node*
__libcpp_db::__find_iterator(const void* __i) const
{
    __i_node* r = nullptr;
    if (__ibeg_ != __iend_)
    {
        size_t h = hash<const void*>()(__i) % (__iend_ - __ibeg_);
        for (__i_node* nd = __ibeg_[h]; nd != nullptr; nd = nd->__next_)
        {
            if (nd->__i_ == __i)
            {
                r = nd;
                break;
            }
        }
    }
    return r;
}

_LIBCPP_HIDDEN
void
__c_node::__remove(__i_node* p)
{
    __i_node** r = find(beg_, end_, p);
    _LIBCPP_ASSERT(r != end_, "debug mode internal logic error __c_node::__remove");
    if (--end_ != r)
        memmove(r, r+1, (end_ - r)*sizeof(__i_node*));
}

_LIBCPP_END_NAMESPACE_STD
