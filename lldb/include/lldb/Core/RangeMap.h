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

namespace lldb_private
{

//----------------------------------------------------------------------
// A simple range class where you get to define the type of the range
// base "B", and the type used for the range byte size "S".
//----------------------------------------------------------------------
template <typename B, typename S> struct Range
{
    typedef B BaseType;
    typedef S SizeType;

    BaseType base;
    SizeType size;

    Range() : base(0), size(0) {}

    Range(BaseType b, SizeType s) : base(b), size(s) {}

    void
    Clear(BaseType b = 0)
    {
        base = b;
        size = 0;
    }

    BaseType
    GetRangeBase() const
    {
        return base;
    }

    void
    SetRangeBase(BaseType b)
    {
        base = b;
    }

    void
    Slide(BaseType slide)
    {
        base += slide;
    }

    BaseType
    GetRangeEnd() const
    {
        return base + size;
    }

    void
    SetRangeEnd(BaseType end)
    {
        if (end > base)
            size = end - base;
        else
            size = 0;
    }

    SizeType
    GetByteSize() const
    {
        return size;
    }

    void
    SetByteSize(SizeType s)
    {
        size = s;
    }

    bool
    IsValid() const
    {
        return size > 0;
    }

    bool
    Contains(BaseType r) const
    {
        return (GetRangeBase() <= r) && (r < GetRangeEnd());
    }

    bool
    ContainsEndInclusive(BaseType r) const
    {
        return (GetRangeBase() <= r) && (r <= GetRangeEnd());
    }

    bool
    Contains(const Range &range) const
    {
        return Contains(range.GetRangeBase()) && ContainsEndInclusive(range.GetRangeEnd());
    }

    // Returns true if the two ranges adjoing or intersect
    bool
    DoesAdjoinOrIntersect(const Range &rhs) const
    {
        return GetRangeBase() <= rhs.GetRangeEnd() && GetRangeEnd() >= rhs.GetRangeBase();
    }

    // Returns true if the two ranges intersect
    bool
    DoesIntersect(const Range &rhs) const
    {
        return GetRangeBase() < rhs.GetRangeEnd() && GetRangeEnd() > rhs.GetRangeBase();
    }

    bool
    operator<(const Range &rhs) const
    {
        return base == rhs.base ? size < rhs.size : base < rhs.base;
    }

    bool
    operator==(const Range &rhs) const
    {
        return base == rhs.base && size == rhs.size;
    }

    bool
    operator!=(const Range &rhs) const
    {
        return !(*this == rhs);
    }
};

//----------------------------------------------------------------------
// A simple range  with data class where you get to define the type of
// the range base "B", the type used for the range byte size "S", and
// the type for the associated data "T".
//----------------------------------------------------------------------
template <typename B, typename S, typename T> struct RangeData : public Range<B, S>
{
    typedef T DataType;

    DataType data;

    RangeData() = default;

    RangeData(B base, S size) : Range<B, S>(base, size), data() {}

    RangeData(B base, S size, DataType d) : Range<B, S>(base, size), data(d) {}

    bool
    operator<(const RangeData &rhs) const
    {
        if (this->base == rhs.base)
            return Range<B, S>::operator<(rhs);

        return this->base < rhs.base;
    }

    bool
    operator==(const RangeData &rhs) const
    {
        return this->data == rhs.data && Range<B, S>::operator==(rhs);
    }

    bool
    operator!=(const RangeData &rhs) const
    {
        return !(*this == rhs);
    }
};

template <typename E, typename C> class RangeVectorBase
{
public:
    typedef E Entry;
    typedef C Collection;
    typedef typename Entry::BaseType BaseType;
    typedef typename Entry::SizeType SizeType;

    void
    Append(const Entry &entry)
    {
        m_entries.push_back(entry);
    }

    bool
    RemoveEntrtAtIndex(uint32_t idx)
    {
        if (idx >= m_entries.size())
            return false;

        m_entries.erase(m_entries.begin() + idx);
        return true;
    }

    void
    Sort()
    {
        std::stable_sort(m_entries.begin(), m_entries.end());
    }

    void
    CombineConsecutiveRanges()
    {
        VerifySorted();

        // Can't combine ranges if we have zero or one range
        if (m_entries.size() <= 1)
            return;

        // First we determine if we can combine any of the Entry objects so we don't end up
        // allocating and making a new collection for no reason
        bool can_combine = false;
        for (auto pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
        {
            if (prev != end && prev->DoesAdjoinOrIntersect(*pos))
            {
                can_combine = true;
                break;
            }
        }

        // We we can combine at least one entry, then we make a new collection and populate it
        // accordingly, and then swap it into place.
        if (can_combine)
        {
            Collection minimal_ranges;
            for (auto pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && prev->DoesAdjoinOrIntersect(*pos))
                    minimal_ranges.back().SetRangeEnd(std::max<BaseType>(prev->GetRangeEnd(), pos->GetRangeEnd()));
                else
                    minimal_ranges.push_back(*pos);
            }

            // Use the swap technique in case our new vector is much smaller. We must swap when
            // using the STL because std::vector objects never release or reduce the memory once it
            // has been allocated/reserved.
            m_entries.swap(minimal_ranges);
        }
    }

    BaseType
    GetMinRangeBase(BaseType fail_value) const
    {
        VerifySorted();

        if (m_entries.empty())
            return fail_value;

        // m_entries must be sorted, so if we aren't empty, we grab the first range's base
        return m_entries.front().GetRangeBase();
    }

    BaseType
    GetMaxRangeEnd(BaseType fail_value) const
    {
        VerifySorted();

        if (m_entries.empty())
            return fail_value;

        // m_entries must be sorted, so if we aren't empty, we grab the last range's end
        return m_entries.back().GetRangeEnd();
    }

    void
    Slide(BaseType slide)
    {
        for (auto pos = m_entries.begin(), end = m_entries.end(); pos != end; ++pos)
            pos->Slide(slide);
    }

    void
    Clear()
    {
        m_entries.clear();
    }

    bool
    IsEmpty() const
    {
        return m_entries.empty();
    }

    size_t
    GetSize() const
    {
        return m_entries.size();
    }

    Entry *
    GetEntryAtIndex(size_t i)
    {
        return i < m_entries.size() ? &m_entries[i] : nullptr;
    }

    const Entry *
    GetEntryAtIndex(size_t i) const
    {
        return i < m_entries.size() ? &m_entries[i] : nullptr;
    }

    // Clients must ensure that "i" is a valid index prior to calling this function
    const Entry &
    GetEntryRef(size_t i) const
    {
        return m_entries[i];
    }

    Entry *
    Back()
    {
        return m_entries.empty() ? nullptr : &m_entries.back();
    }

    const Entry *
    Back() const
    {
        return m_entries.empty() ? nullptr : &m_entries.back();
    }

    uint32_t
    FindEntryIndexThatContains(const Entry &range) const
    {
        VerifySorted();

        if (m_entries.empty())
            return UINT32_MAX;

        auto begin = m_entries.begin(), end = m_entries.end(), pos = std::lower_bound(begin, end, range, BaseLessThan);
        if (pos != end && pos->Contains(range))
            return std::distance(begin, pos);

        if (pos != begin)
        {
            --pos;
            if (pos->Contains(range))
                return std::distance(begin, pos);
        }
        return UINT32_MAX;
    }

    uint32_t
    FindEntryIndexThatContains(BaseType addr) const
    {
        return FindEntryIndexThatContains(Entry(addr, 1));
    }

    Entry *
    FindEntryThatContains(BaseType addr)
    {
        return GetEntryAtIndex(FindEntryIndexThatContains(addr));
    }

    const Entry *
    FindEntryThatContains(BaseType addr) const
    {
        return GetEntryAtIndex(FindEntryIndexThatContains(addr));
    }

    const Entry *
    FindEntryThatContains(const Entry &range) const
    {
        return GetEntryAtIndex(FindEntryIndexThatContains(range));
    }

    void
    Reserve(size_t size)
    {
        m_entries.resize(size);
    }

    // Calculate the byte size of ranges with zero byte sizes by finding the next entry with a
    // base address > the current base address
    void
    CalculateSizesOfZeroByteSizeRanges()
    {
        VerifySorted();

        for (auto pos = m_entries.begin(), end = m_entries.end(); pos != end; ++pos)
        {
            if (pos->GetByteSize() == 0)
            {
                // Watch out for multiple entries with same address and make sure we find an entry
                //that is greater than the current base address before we use that for the size
                auto curr_base = pos->GetRangeBase();
                for (auto next = pos + 1; next != end; ++next)
                {
                    auto next_base = next->GetRangeBase();
                    if (next_base > curr_base)
                    {
                        pos->SetByteSize(next_base - curr_base);
                        break;
                    }
                }
            }
        }
    }

    uint32_t
    FindEntryIndexesThatContain(BaseType addr, std::vector<uint32_t> &indexes) const
    {
        VerifySorted();

        if (!m_entries.empty())
        {
            for (const auto &entry : m_entries)
            {
                if (entry.Contains(addr))
                    indexes.push_back(entry.data);
            }
        }
        return indexes.size();
    }

    static bool
    BaseLessThan(const Entry &lhs, const Entry &rhs)
    {
        return lhs.GetRangeBase() < rhs.GetRangeBase();
    }

protected:
    void
    VerifySorted() const
    {
#ifdef ASSERT_RANGEMAP_ARE_SORTED
        for (auto pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            assert(prev == end || *pos >= *prev);
#endif
    }

    Collection m_entries;
};

template <typename E, typename S> class RangeDataVectorBase : public RangeVectorBase<E, S>
{
public:
    void
    CombineConsecutiveEntriesWithEqualData()
    {
        VerifySorted();

        // First we determine if we can combine any of the Entry objects so we
        // don't end up allocating and making a new collection for no reason
        bool can_combine = false;
        for (auto pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
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
            for (auto pos = m_entries.begin(), end = m_entries.end(), prev = end; pos != end; prev = pos++)
            {
                if (prev != end && prev->data == pos->data)
                    minimal_ranges.back().SetRangeEnd(pos->GetRangeEnd());
                else
                    minimal_ranges.push_back(*pos);
            }

            // Use the swap technique in case our new vector is much smaller.
            // We must swap when using the STL because std::vector objects never
            // release or reduce the memory once it has been allocated/reserved.
            m_entries.swap(minimal_ranges);
        }
    }

protected:
    using typename RangeVectorBase<E, S>::Collection;
    using RangeVectorBase<E, S>::VerifySorted;
    using RangeVectorBase<E, S>::m_entries;
};

// Use public inheritance to define these types instead of alias templates because MSVC 2013
// generates incorrect code for alias templates.

template <typename B, typename S> class RangeVector : public RangeVectorBase<Range<B, S>, std::vector<Range<B, S>>>
{
};

template <typename B, typename S, size_t N>
class RangeArray : public RangeVectorBase<Range<B, S>, llvm::SmallVector<Range<B, S>, N>>
{
};

template <typename B, typename S, typename T>
class RangeDataVector : public RangeDataVectorBase<RangeData<B, S, T>, std::vector<RangeData<B, S, T>>>
{
};

template <typename B, typename S, typename T, size_t N>
class RangeDataArray : public RangeDataVectorBase<RangeData<B, S, T>, llvm::SmallVector<RangeData<B, S, T>, N>>
{
};

} // namespace lldb_private

#endif // liblldb_RangeMap_h_
