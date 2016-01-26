//===-- RangeMap.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RangeMap_h_
#define liblldb_RangeMap_h_

// C Includes
// C++ Includes
#include <algorithm>
#include <vector>

// Other libraries and framework includes
#include "llvm/ADT/SmallVector.h"

// Project includes
#include "lldb/lldb-private.h"

// Uncomment to make sure all Range objects are sorted when needed
//#define ASSERT_RANGEMAP_ARE_SORTED

namespace lldb_private {
    
    //----------------------------------------------------------------------
    // Templatized classes for dealing with generic ranges and also
    // collections of ranges, or collections of ranges that have associated
    // data.
    //----------------------------------------------------------------------
    
    //----------------------------------------------------------------------
    // A simple range class where you get to define the type of the range
    // base "B", and the type used for the range byte size "S".
    //----------------------------------------------------------------------
    template <typename B, typename S>
    struct Range
    {
        typedef B BaseType;
        typedef S SizeType;

        BaseType base;
        SizeType size;
        
        Range () :
            base (0),
            size (0)
        {
        }
        
        Range (BaseType b, SizeType s) :
            base (b),
            size (s)
        {
        }
        
        void
        Clear (BaseType b = 0)
        {
            base = b;
            size = 0;
        }

        // Set the start value for the range, and keep the same size
        BaseType
        GetRangeBase () const
        {
            return base;
        }
        
        void
        SetRangeBase (BaseType b)
        {
            base = b;
        }
        
        void
        Slide (BaseType slide)
        {
            base += slide;
        }
        
        BaseType
        GetRangeEnd () const
        {
            return base + size;
        }
        
        void
        SetRangeEnd (BaseType end)
        {
            if (end > base)
                size = end - base;
            else
                size = 0;
        }
        
        SizeType
        GetByteSize () const
        {
            return size;
        }
        
        void
        SetByteSize (SizeType s)
        {
            size = s;
        }
        
        bool
        IsValid() const
        {
            return size > 0;
        }
        
        bool
        Contains (BaseType r) const
        {
            return (GetRangeBase() <= r) && (r < GetRangeEnd());
        }
        
        bool
        ContainsEndInclusive (BaseType r) const
        {
            return (GetRangeBase() <= r) && (r <= GetRangeEnd());
        }
        
        bool 
        Contains (const Range& range) const
        {
            return Contains(range.GetRangeBase()) && ContainsEndInclusive(range.GetRangeEnd());
        }

        // Returns true if the two ranges adjoing or intersect
        bool
        DoesAdjoinOrIntersect (const Range &rhs) const
        {
            const BaseType lhs_base = this->GetRangeBase();
            const BaseType rhs_base = rhs.GetRangeBase();
            const BaseType lhs_end = this->GetRangeEnd();
            const BaseType rhs_end = rhs.GetRangeEnd();
            bool result = (lhs_base <= rhs_end) && (lhs_end >= rhs_base);
            return result;
        }

        // Returns true if the two ranges intersect
        bool
        DoesIntersect (const Range &rhs) const
        {
            const BaseType lhs_base = this->GetRangeBase();
            const BaseType rhs_base = rhs.GetRangeBase();
            const BaseType lhs_end = this->GetRangeEnd();
            const BaseType rhs_end = rhs.GetRangeEnd();
            bool result = (lhs_base < rhs_end) && (lhs_end > rhs_base);
            return result;
        }

        bool
        operator < (const Range &rhs) const
        {
            if (base == rhs.base)
                return size < rhs.size;
            return base < rhs.base;
        }
        
        bool
        operator == (const Range &rhs) const
        {
            return base == rhs.base && size == rhs.size;
        }
        
        bool
        operator != (const Range &rhs) const
        {
            return  base != rhs.base || size != rhs.size;
        }
    };
    
    //----------------------------------------------------------------------
    // A range array class where you get to define the type of the ranges
    // that the collection contains.
    //----------------------------------------------------------------------

    template <typename B, typename S, unsigned N>
    class RangeArray
    {
    public:
        typedef B BaseType;
        typedef S SizeType;
        typedef Range<B,S> Entry;
        typedef llvm::SmallVector<Entry, N> Collection;

        RangeArray() = default;

        ~RangeArray() = default;

        void
        Append (const Entry &entry)
        {
            m_entries.push_back (entry);
        }
        
        bool
        RemoveEntrtAtIndex (uint32_t idx)
        {
            if (idx < m_entries.size())
            {
                m_entries.erase (m_entries.begin() + idx);
                return true;
            }
            return false;
        }
        
        void
        Sort ()
        {
            if (m_entries.size() > 1)
                std::stable_sort (m_entries.begin(), m_entries.end());
        }
        
#ifdef ASSERT_RANGEMAP_ARE_SORTED
        bool
        IsSorted () const
        {
            typename Collection::const_iterator pos, end, prev;
            // First we determine if we can combine any of the Entry objects so we
            // don't end up allocating and making a new collection for no reason
            for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && *pos < *prev)
                    return false;
            }
            return true;
        }
#endif        

        void
        CombineConsecutiveRanges ()
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            // Can't combine if ranges if we have zero or one range
            if (m_entries.size() > 1)
            {
                // The list should be sorted prior to calling this function
                typename Collection::iterator pos;
                typename Collection::iterator end;
                typename Collection::iterator prev;
                bool can_combine = false;
                // First we determine if we can combine any of the Entry objects so we
                // don't end up allocating and making a new collection for no reason
                for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
                {
                    if (prev != end && prev->DoesAdjoinOrIntersect(*pos))
                    {
                        can_combine = true;
                        break;
                    }
                }
                
                // We we can combine at least one entry, then we make a new collection
                // and populate it accordingly, and then swap it into place. 
                if (can_combine)
                {
                    Collection minimal_ranges;
                    for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
                    {
                        if (prev != end && prev->DoesAdjoinOrIntersect(*pos))
                            minimal_ranges.back().SetRangeEnd (std::max<BaseType>(prev->GetRangeEnd(), pos->GetRangeEnd()));
                        else
                            minimal_ranges.push_back (*pos);
                    }
                    // Use the swap technique in case our new vector is much smaller.
                    // We must swap when using the STL because std::vector objects never
                    // release or reduce the memory once it has been allocated/reserved.
                    m_entries.swap (minimal_ranges);
                }
            }
        }

        BaseType
        GetMinRangeBase (BaseType fail_value) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (m_entries.empty())
                return fail_value;
            // m_entries must be sorted, so if we aren't empty, we grab the
            // first range's base
            return m_entries.front().GetRangeBase();
        }

        BaseType
        GetMaxRangeEnd (BaseType fail_value) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (m_entries.empty())
                return fail_value;
            // m_entries must be sorted, so if we aren't empty, we grab the
            // last range's end
            return m_entries.back().GetRangeEnd();
        }
        
        void
        Slide (BaseType slide)
        {
            typename Collection::iterator pos, end;
            for (pos = m_entries.begin(), end = m_entries.end(); pos != end; ++pos)
                pos->Slide (slide);
        }
        
        void
        Clear ()
        {
            m_entries.clear();
        }
        
        bool
        IsEmpty () const
        {
            return m_entries.empty();
        }
        
        size_t
        GetSize () const
        {
            return m_entries.size();
        }
        
        const Entry *
        GetEntryAtIndex (size_t i) const
        {
            return ((i < m_entries.size()) ? &m_entries[i] : nullptr);
        }
        
        // Clients must ensure that "i" is a valid index prior to calling this function
        const Entry &
        GetEntryRef (size_t i) const
        {
            return m_entries[i];
        }

        Entry *
        Back()
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }

        const Entry *
        Back() const
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }

        static bool 
        BaseLessThan (const Entry& lhs, const Entry& rhs)
        {
            return lhs.GetRangeBase() < rhs.GetRangeBase();
        }
        
        uint32_t
        FindEntryIndexThatContains (B addr) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (!m_entries.empty())
            {
                Entry entry (addr, 1);
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                if (pos != end && pos->Contains(addr))
                {
                    return std::distance (begin, pos);
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(addr))
                        return std::distance (begin, pos);
                }
            }
            return UINT32_MAX;
        }

        const Entry *
        FindEntryThatContains (B addr) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (!m_entries.empty())
            {
                Entry entry (addr, 1);
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                if (pos != end && pos->Contains(addr))
                {
                    return &(*pos); 
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(addr))
                    {
                        return &(*pos); 
                    }
                }
            }
            return nullptr;
        }

        const Entry *
        FindEntryThatContains (const Entry &range) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (!m_entries.empty())
            {
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, range, BaseLessThan);
                
                if (pos != end && pos->Contains(range))
                {
                    return &(*pos); 
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(range))
                    {
                        return &(*pos); 
                    }
                }
            }
            return nullptr;
        }

    protected:
        Collection m_entries;
    };

    template <typename B, typename S>
    class RangeVector
    {
    public:
        typedef B BaseType;
        typedef S SizeType;
        typedef Range<B,S> Entry;
        typedef std::vector<Entry> Collection;

        RangeVector() = default;

        ~RangeVector() = default;

        void
        Append (const Entry &entry)
        {
            m_entries.push_back (entry);
        }
        
        bool
        RemoveEntrtAtIndex (uint32_t idx)
        {
            if (idx < m_entries.size())
            {
                m_entries.erase (m_entries.begin() + idx);
                return true;
            }
            return false;
        }
        
        void
        Sort ()
        {
            if (m_entries.size() > 1)
                std::stable_sort (m_entries.begin(), m_entries.end());
        }
        
#ifdef ASSERT_RANGEMAP_ARE_SORTED
        bool
        IsSorted () const
        {
            typename Collection::const_iterator pos, end, prev;
            // First we determine if we can combine any of the Entry objects so we
            // don't end up allocating and making a new collection for no reason
            for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && *pos < *prev)
                    return false;
            }
            return true;
        }
#endif

        void
        CombineConsecutiveRanges ()
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            // Can't combine if ranges if we have zero or one range
            if (m_entries.size() > 1)
            {
                // The list should be sorted prior to calling this function
                typename Collection::iterator pos;
                typename Collection::iterator end;
                typename Collection::iterator prev;
                bool can_combine = false;
                // First we determine if we can combine any of the Entry objects so we
                // don't end up allocating and making a new collection for no reason
                for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
                {
                    if (prev != end && prev->DoesAdjoinOrIntersect(*pos))
                    {
                        can_combine = true;
                        break;
                    }
                }
                
                // We we can combine at least one entry, then we make a new collection
                // and populate it accordingly, and then swap it into place.
                if (can_combine)
                {
                    Collection minimal_ranges;
                    for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
                    {
                        if (prev != end && prev->DoesAdjoinOrIntersect(*pos))
                            minimal_ranges.back().SetRangeEnd (std::max<BaseType>(prev->GetRangeEnd(), pos->GetRangeEnd()));
                        else
                            minimal_ranges.push_back (*pos);
                    }
                    // Use the swap technique in case our new vector is much smaller.
                    // We must swap when using the STL because std::vector objects never
                    // release or reduce the memory once it has been allocated/reserved.
                    m_entries.swap (minimal_ranges);
                }
            }
        }

        BaseType
        GetMinRangeBase (BaseType fail_value) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (m_entries.empty())
                return fail_value;
            // m_entries must be sorted, so if we aren't empty, we grab the
            // first range's base
            return m_entries.front().GetRangeBase();
        }
        
        BaseType
        GetMaxRangeEnd (BaseType fail_value) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (m_entries.empty())
                return fail_value;
            // m_entries must be sorted, so if we aren't empty, we grab the
            // last range's end
            return m_entries.back().GetRangeEnd();
        }
        
        void
        Slide (BaseType slide)
        {
            typename Collection::iterator pos, end;
            for (pos = m_entries.begin(), end = m_entries.end(); pos != end; ++pos)
                pos->Slide (slide);
        }
        
        void
        Clear ()
        {
            m_entries.clear();
        }

        void
        Reserve (typename Collection::size_type size)
        {
            m_entries.reserve (size);
        }

        bool
        IsEmpty () const
        {
            return m_entries.empty();
        }
        
        size_t
        GetSize () const
        {
            return m_entries.size();
        }
        
        const Entry *
        GetEntryAtIndex (size_t i) const
        {
            return ((i < m_entries.size()) ? &m_entries[i] : nullptr);
        }
        
        // Clients must ensure that "i" is a valid index prior to calling this function
        const Entry &
        GetEntryRef (size_t i) const
        {
            return m_entries[i];
        }
        
        Entry *
        Back()
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }
        
        const Entry *
        Back() const
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }
        
        static bool
        BaseLessThan (const Entry& lhs, const Entry& rhs)
        {
            return lhs.GetRangeBase() < rhs.GetRangeBase();
        }
        
        uint32_t
        FindEntryIndexThatContains (B addr) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (!m_entries.empty())
            {
                Entry entry (addr, 1);
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                if (pos != end && pos->Contains(addr))
                {
                    return std::distance (begin, pos);
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(addr))
                        return std::distance (begin, pos);
                }
            }
            return UINT32_MAX;
        }
        
        const Entry *
        FindEntryThatContains (B addr) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (!m_entries.empty())
            {
                Entry entry (addr, 1);
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                if (pos != end && pos->Contains(addr))
                {
                    return &(*pos);
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(addr))
                    {
                        return &(*pos);
                    }
                }
            }
            return nullptr;
        }
        
        const Entry *
        FindEntryThatContains (const Entry &range) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if (!m_entries.empty())
            {
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, range, BaseLessThan);
                
                if (pos != end && pos->Contains(range))
                {
                    return &(*pos);
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(range))
                    {
                        return &(*pos);
                    }
                }
            }
            return nullptr;
        }
        
    protected:
        Collection m_entries;
    };

    //----------------------------------------------------------------------
    // A simple range  with data class where you get to define the type of
    // the range base "B", the type used for the range byte size "S", and
    // the type for the associated data "T".
    //----------------------------------------------------------------------
    template <typename B, typename S, typename T>
    struct RangeData : public Range<B,S>
    {
        typedef T DataType;
        
        DataType data;
        
        RangeData () :
            Range<B,S> (),
            data ()
        {
        }
        
        RangeData (B base, S size) :
            Range<B,S> (base, size),
            data ()
        {
        }
        
        RangeData (B base, S size, DataType d) :
            Range<B,S> (base, size),
            data (d)
        {
        }
        
        bool
        operator < (const RangeData &rhs) const
        {
            if (this->base == rhs.base)
            {
                if (this->size == rhs.size)
                    return this->data < rhs.data;
                else
                    return this->size < rhs.size;
            }
            return this->base < rhs.base;
        }
        
        bool
        operator == (const RangeData &rhs) const
        {
            return this->GetRangeBase() == rhs.GetRangeBase() &&
                   this->GetByteSize() == rhs.GetByteSize() &&
                   this->data      == rhs.data;
        }
        
        bool
        operator != (const RangeData &rhs) const
        {
            return this->GetRangeBase() != rhs.GetRangeBase() ||
                   this->GetByteSize() != rhs.GetByteSize() ||
                   this->data      != rhs.data;
        }
    };
    
    template <typename B, typename S, typename T, unsigned N>
    class RangeDataArray
    {
    public:
        typedef RangeData<B,S,T> Entry;
        typedef llvm::SmallVector<Entry, N> Collection;

        RangeDataArray() = default;

        ~RangeDataArray() = default;

        void
        Append (const Entry &entry)
        {
            m_entries.push_back (entry);
        }
    
        void
        Sort ()
        {
            if (m_entries.size() > 1)
                std::stable_sort (m_entries.begin(), m_entries.end());
        }
    
#ifdef ASSERT_RANGEMAP_ARE_SORTED
        bool
        IsSorted () const
        {
            typename Collection::const_iterator pos, end, prev;
            // First we determine if we can combine any of the Entry objects so we
            // don't end up allocating and making a new collection for no reason
            for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && *pos < *prev)
                    return false;
            }
            return true;
        }
#endif

        void
        CombineConsecutiveEntriesWithEqualData ()
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            typename Collection::iterator pos;
            typename Collection::iterator end;
            typename Collection::iterator prev;
            bool can_combine = false;
            // First we determine if we can combine any of the Entry objects so we
            // don't end up allocating and making a new collection for no reason
            for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && prev->data == pos->data)
                {
                    can_combine = true;
                    break;
                }
            }
            
            // We we can combine at least one entry, then we make a new collection
            // and populate it accordingly, and then swap it into place. 
            if (can_combine)
            {
                Collection minimal_ranges;
                for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
                {
                    if (prev != end && prev->data == pos->data)
                        minimal_ranges.back().SetRangeEnd (pos->GetRangeEnd());
                    else
                        minimal_ranges.push_back (*pos);
                }
                // Use the swap technique in case our new vector is much smaller.
                // We must swap when using the STL because std::vector objects never
                // release or reduce the memory once it has been allocated/reserved.
                m_entries.swap (minimal_ranges);
            }
        }
    
        void
        Clear ()
        {
            m_entries.clear();
        }
        
        bool
        IsEmpty () const
        {
            return m_entries.empty();
        }
        
        size_t
        GetSize () const
        {
            return m_entries.size();
        }
        
        const Entry *
        GetEntryAtIndex (size_t i) const
        {
            return ((i < m_entries.size()) ? &m_entries[i] : nullptr);
        }

        // Clients must ensure that "i" is a valid index prior to calling this function
        const Entry &
        GetEntryRef (size_t i) const
        {
            return m_entries[i];
        }

        static bool
        BaseLessThan (const Entry& lhs, const Entry& rhs)
        {
            return lhs.GetRangeBase() < rhs.GetRangeBase();
        }
        
        uint32_t
        FindEntryIndexThatContains (B addr) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if ( !m_entries.empty() )
            {
                Entry entry (addr, 1);
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                if (pos != end && pos->Contains(addr))
                {
                    return std::distance (begin, pos);
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(addr))
                        return std::distance (begin, pos);
                }
            }
            return UINT32_MAX;
        }

        Entry *
        FindEntryThatContains (B addr)
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if ( !m_entries.empty() )
            {
                Entry entry;
                entry.SetRangeBase(addr);
                entry.SetByteSize(1);
                typename Collection::iterator begin = m_entries.begin();
                typename Collection::iterator end = m_entries.end();
                typename Collection::iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                if (pos != end && pos->Contains(addr))
                {
                    return &(*pos); 
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(addr))
                    {
                        return &(*pos); 
                    }
                }
            }
            return nullptr;
        }

        const Entry *
        FindEntryThatContains (B addr) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if ( !m_entries.empty() )
            {
                Entry entry;
                entry.SetRangeBase(addr);
                entry.SetByteSize(1);
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                if (pos != end && pos->Contains(addr))
                {
                    return &(*pos); 
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(addr))
                    {
                        return &(*pos); 
                    }
                }
            }
            return nullptr;
        }
        
        const Entry *
        FindEntryThatContains (const Entry &range) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if ( !m_entries.empty() )
            {
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, range, BaseLessThan);
                
                if (pos != end && pos->Contains(range))
                {
                    return &(*pos); 
                }
                else if (pos != begin)
                {
                    --pos;
                    if (pos->Contains(range))
                    {
                        return &(*pos); 
                    }
                }
            }
            return nullptr;
        }
        
        Entry *
        Back()
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }

        const Entry *
        Back() const
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }

    protected:
        Collection m_entries;
    };

    // Same as RangeDataArray, but uses std::vector as to not
    // require static storage of N items in the class itself
    template <typename B, typename S, typename T>
    class RangeDataVector
    {
    public:
        typedef RangeData<B,S,T> Entry;
        typedef std::vector<Entry> Collection;

        RangeDataVector() = default;

        ~RangeDataVector() = default;

        void
        Append (const Entry &entry)
        {
            m_entries.push_back (entry);
        }
        
        void
        Sort ()
        {
            if (m_entries.size() > 1)
                std::stable_sort (m_entries.begin(), m_entries.end());
        }
        
#ifdef ASSERT_RANGEMAP_ARE_SORTED
        bool
        IsSorted () const
        {
            typename Collection::const_iterator pos, end, prev;
            // First we determine if we can combine any of the Entry objects so we
            // don't end up allocating and making a new collection for no reason
            for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && *pos < *prev)
                    return false;
            }
            return true;
        }
#endif
        
        void
        CombineConsecutiveEntriesWithEqualData ()
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            typename Collection::iterator pos;
            typename Collection::iterator end;
            typename Collection::iterator prev;
            bool can_combine = false;
            // First we determine if we can combine any of the Entry objects so we
            // don't end up allocating and making a new collection for no reason
            for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && prev->data == pos->data)
                {
                    can_combine = true;
                    break;
                }
            }
            
            // We we can combine at least one entry, then we make a new collection
            // and populate it accordingly, and then swap it into place.
            if (can_combine)
            {
                Collection minimal_ranges;
                for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
                {
                    if (prev != end && prev->data == pos->data)
                        minimal_ranges.back().SetRangeEnd (pos->GetRangeEnd());
                    else
                        minimal_ranges.push_back (*pos);
                }
                // Use the swap technique in case our new vector is much smaller.
                // We must swap when using the STL because std::vector objects never
                // release or reduce the memory once it has been allocated/reserved.
                m_entries.swap (minimal_ranges);
            }
        }
        
        // Calculate the byte size of ranges with zero byte sizes by finding
        // the next entry with a base address > the current base address
        void
        CalculateSizesOfZeroByteSizeRanges ()
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            typename Collection::iterator pos;
            typename Collection::iterator end;
            typename Collection::iterator next;
            for (pos = m_entries.begin(), end = m_entries.end(); pos != end; ++pos)
            {
                if (pos->GetByteSize() == 0)
                {
                    // Watch out for multiple entries with same address and make sure
                    // we find an entry that is greater than the current base address
                    // before we use that for the size
                    auto curr_base = pos->GetRangeBase();
                    for (next = pos + 1; next != end; ++next)
                    {
                        auto next_base = next->GetRangeBase();
                        if (next_base > curr_base)
                        {
                            pos->SetByteSize (next_base - curr_base);
                            break;
                        }
                    }
                }
            }
        }
        
        void
        Clear ()
        {
            m_entries.clear();
        }

        void
        Reserve (typename Collection::size_type size)
        {
            m_entries.resize (size);
        }

        bool
        IsEmpty () const
        {
            return m_entries.empty();
        }
        
        size_t
        GetSize () const
        {
            return m_entries.size();
        }
        
        const Entry *
        GetEntryAtIndex (size_t i) const
        {
            return ((i < m_entries.size()) ? &m_entries[i] : nullptr);
        }
        
        // Clients must ensure that "i" is a valid index prior to calling this function
        const Entry &
        GetEntryRef (size_t i) const
        {
            return m_entries[i];
        }
        
        static bool
        BaseLessThan (const Entry& lhs, const Entry& rhs)
        {
            return lhs.GetRangeBase() < rhs.GetRangeBase();
        }
        
        uint32_t
        FindEntryIndexThatContains (B addr) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if ( !m_entries.empty() )
            {
                Entry entry (addr, 1);
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                while(pos != begin && pos[-1].Contains(addr))
                    --pos;

                if (pos != end && pos->Contains(addr))
                    return std::distance (begin, pos);
            }
            return UINT32_MAX;
        }

        uint32_t
        FindEntryIndexesThatContain(B addr, std::vector<uint32_t> &indexes) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif

            if (!m_entries.empty())
            {
                typename Collection::const_iterator pos;
                for (const auto &entry : m_entries)
                {
                    if (entry.Contains(addr))
                        indexes.push_back(entry.data);
                }
            }
            return indexes.size() ;
        }
        
        Entry *
        FindEntryThatContains (B addr)
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if ( !m_entries.empty() )
            {
                Entry entry;
                entry.SetRangeBase(addr);
                entry.SetByteSize(1);
                typename Collection::iterator begin = m_entries.begin();
                typename Collection::iterator end = m_entries.end();
                typename Collection::iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);

                while(pos != begin && pos[-1].Contains(addr))
                    --pos;
                
                if (pos != end && pos->Contains(addr))
                    return &(*pos);
            }
            return nullptr;
        }

        const Entry *
        FindEntryThatContains (B addr) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if ( !m_entries.empty() )
            {
                Entry entry;
                entry.SetRangeBase(addr);
                entry.SetByteSize(1);
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                while(pos != begin && pos[-1].Contains(addr))
                    --pos;

                if (pos != end && pos->Contains(addr))
                    return &(*pos);
            }
            return nullptr;
        }
        
        const Entry *
        FindEntryThatContains (const Entry &range) const
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if ( !m_entries.empty() )
            {
                typename Collection::const_iterator begin = m_entries.begin();
                typename Collection::const_iterator end = m_entries.end();
                typename Collection::const_iterator pos = std::lower_bound (begin, end, range, BaseLessThan);
                
                while(pos != begin && pos[-1].Contains(range))
                    --pos;

                if (pos != end && pos->Contains(range))
                    return &(*pos);
            }
            return nullptr;
        }
        
        Entry *
        Back()
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }
        
        const Entry *
        Back() const
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }
        
    protected:
        Collection m_entries;
    };

    //----------------------------------------------------------------------
    // A simple range  with data class where you get to define the type of
    // the range base "B", the type used for the range byte size "S", and
    // the type for the associated data "T".
    //----------------------------------------------------------------------
    template <typename B, typename T>
    struct AddressData
    {
        typedef B BaseType;
        typedef T DataType;
        
        BaseType addr;
        DataType data;
        
        AddressData () :
            addr (),
            data ()
        {
        }
        
        AddressData (B a, DataType d) :
            addr (a),
            data (d)
        {
        }
        
        bool
        operator < (const AddressData &rhs) const
        {
            if (this->addr == rhs.addr)
                return this->data < rhs.data;
            return this->addr < rhs.addr;
        }
        
        bool
        operator == (const AddressData &rhs) const
        {
            return this->addr == rhs.addr &&
                   this->data == rhs.data;
        }
        
        bool
        operator != (const AddressData &rhs) const
        {
            return this->addr != rhs.addr ||
                   this->data == rhs.data;
        }
    };

    template <typename B, typename T, unsigned N>
    class AddressDataArray
    {
    public:
        typedef AddressData<B,T> Entry;
        typedef llvm::SmallVector<Entry, N> Collection;

        AddressDataArray() = default;

        ~AddressDataArray() = default;

        void
        Append (const Entry &entry)
        {
            m_entries.push_back (entry);
        }
    
        void
        Sort ()
        {
            if (m_entries.size() > 1)
                std::stable_sort (m_entries.begin(), m_entries.end());
        }
    
#ifdef ASSERT_RANGEMAP_ARE_SORTED
        bool
        IsSorted () const
        {
            typename Collection::const_iterator pos, end, prev;
            // First we determine if we can combine any of the Entry objects so we
            // don't end up allocating and making a new collection for no reason
            for (pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && *pos < *prev)
                    return false;
            }
            return true;
        }
#endif

        void
        Clear ()
        {
            m_entries.clear();
        }
        
        bool
        IsEmpty () const
        {
            return m_entries.empty();
        }
        
        size_t
        GetSize () const
        {
            return m_entries.size();
        }
        
        const Entry *
        GetEntryAtIndex (size_t i) const
        {
            return ((i < m_entries.size()) ? &m_entries[i] : nullptr);
        }

        // Clients must ensure that "i" is a valid index prior to calling this function
        const Entry &
        GetEntryRef (size_t i) const
        {
            return m_entries[i];
        }

        static bool 
        BaseLessThan (const Entry& lhs, const Entry& rhs)
        {
            return lhs.addr < rhs.addr;
        }
        
        Entry *
        FindEntry (B addr, bool exact_match_only)
        {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
            assert (IsSorted());
#endif
            if ( !m_entries.empty() )
            {
                Entry entry;
                entry.addr = addr;
                typename Collection::iterator begin = m_entries.begin();
                typename Collection::iterator end = m_entries.end();
                typename Collection::iterator pos = std::lower_bound (begin, end, entry, BaseLessThan);
                
                while(pos != begin && pos[-1].addr == addr)
                    --pos;

                if (pos != end)
                {
                    if (pos->addr == addr || !exact_match_only)
                        return &(*pos);
                }
            }
            return nullptr;
        }
        
        const Entry *
        FindNextEntry (const Entry *entry)
        {
            if (entry >= &*m_entries.begin() && entry + 1 < &*m_entries.end())
                return entry + 1;
            return nullptr;
        }
        
        Entry *
        Back()
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }

        const Entry *
        Back() const
        {
            return (m_entries.empty() ? nullptr : &m_entries.back());
        }

    protected:
        Collection m_entries;
    };

} // namespace lldb_private

#endif // liblldb_RangeMap_h_
