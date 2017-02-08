//===--------------------- filesystem/path.cpp ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "experimental/filesystem"
#include "string_view"
#include "utility"

namespace { namespace parser
{
using namespace std;
using namespace std::experimental::filesystem;

using string_view_t = path::__string_view;
using string_view_pair = pair<string_view_t, string_view_t>;
using PosPtr = path::value_type const*;

struct PathParser {
  enum ParserState : unsigned char {
    // Zero is a special sentinel value used by default constructed iterators.
    PS_BeforeBegin = 1,
    PS_InRootName,
    PS_InRootDir,
    PS_InFilenames,
    PS_InTrailingSep,
    PS_AtEnd
  };

  const string_view_t Path;
  string_view_t RawEntry;
  ParserState State;

private:
  PathParser(string_view_t P, ParserState State) noexcept
      : Path(P), State(State) {}

public:
  PathParser(string_view_t P, string_view_t E, unsigned char S)
      : Path(P), RawEntry(E), State(static_cast<ParserState>(S)) {
    // S cannot be '0' or PS_BeforeBegin.
  }

  static PathParser CreateBegin(string_view_t P) noexcept {
    PathParser PP(P, PS_BeforeBegin);
    PP.increment();
    return PP;
  }

  static PathParser CreateEnd(string_view_t P) noexcept {
    PathParser PP(P, PS_AtEnd);
    return PP;
  }

  PosPtr peek() const noexcept {
    auto TkEnd = getNextTokenStartPos();
    auto End = getAfterBack();
    return TkEnd == End ? nullptr : TkEnd;
  }

  void increment() noexcept {
    const PosPtr End = getAfterBack();
    const PosPtr Start = getNextTokenStartPos();
    if (Start == End)
      return makeState(PS_AtEnd);

    switch (State) {
    case PS_BeforeBegin: {
      PosPtr TkEnd = consumeSeparator(Start, End);
      // If we consumed exactly two separators we have a root name.
      if (TkEnd && TkEnd == Start + 2) {
        // FIXME Do we need to consume a name or is '//' a root name on its own?
        // what about '//.', '//..', '//...'?
        auto NameEnd = consumeName(TkEnd, End);
        if (NameEnd)
          TkEnd = NameEnd;
        return makeState(PS_InRootName, Start, TkEnd);
      }
      else if (TkEnd)
        return makeState(PS_InRootDir, Start, TkEnd);
      else
        return makeState(PS_InFilenames, Start, consumeName(Start, End));
    }

    case PS_InRootName:
      return makeState(PS_InRootDir, Start, consumeSeparator(Start, End));
    case PS_InRootDir:
      return makeState(PS_InFilenames, Start, consumeName(Start, End));

    case PS_InFilenames: {
      PosPtr SepEnd = consumeSeparator(Start, End);
      if (SepEnd != End) {
        PosPtr TkEnd = consumeName(SepEnd, End);
        if (TkEnd)
          return makeState(PS_InFilenames, SepEnd, TkEnd);
      }
      return makeState(PS_InTrailingSep, Start, SepEnd);
    }

    case PS_InTrailingSep:
      return makeState(PS_AtEnd);

    case PS_AtEnd:
      _LIBCPP_UNREACHABLE();
    }
  }

  void decrement() noexcept {
    const PosPtr REnd = getBeforeFront();
    const PosPtr RStart = getCurrentTokenStartPos() - 1;

    switch (State) {
    case PS_AtEnd: {
      // Try to consume a trailing separator or root directory first.
      if (PosPtr SepEnd = consumeSeparator(RStart, REnd)) {
        if (SepEnd == REnd)
          return makeState((RStart == REnd + 2) ? PS_InRootName : PS_InRootDir,
                           Path.data(), RStart + 1);
        // Check if we're seeing the root directory separator
        auto PP = CreateBegin(Path);
        bool InRootDir = PP.State == PS_InRootName &&
            &PP.RawEntry.back() == SepEnd;
        return makeState(InRootDir ? PS_InRootDir : PS_InTrailingSep,
                         SepEnd + 1, RStart + 1);
      } else {
        PosPtr TkStart = consumeName(RStart, REnd);
        if (TkStart == REnd + 2 && consumeSeparator(TkStart, REnd) == REnd)
          return makeState(PS_InRootName, Path.data(), RStart + 1);
        else
          return makeState(PS_InFilenames, TkStart + 1, RStart + 1);
      }
    }
    case PS_InTrailingSep:
      return makeState(PS_InFilenames, consumeName(RStart, REnd) + 1, RStart + 1);
    case PS_InFilenames: {
      PosPtr SepEnd = consumeSeparator(RStart, REnd);
      if (SepEnd == REnd)
        return makeState((RStart == REnd + 2) ? PS_InRootName : PS_InRootDir,
                         Path.data(), RStart + 1);
      PosPtr TkEnd = consumeName(SepEnd, REnd);
      if (TkEnd == REnd + 2 && consumeSeparator(TkEnd, REnd) == REnd)
        return makeState(PS_InRootDir, SepEnd + 1, RStart + 1);
      return makeState(PS_InFilenames, TkEnd + 1, SepEnd + 1);
    }
    case PS_InRootDir:
      return makeState(PS_InRootName, Path.data(), RStart + 1);
    case PS_InRootName:
    case PS_BeforeBegin:
      _LIBCPP_UNREACHABLE();
    }
  }

  /// \brief Return a view with the "preferred representation" of the current
  ///   element. For example trailing separators are represented as a '.'
  string_view_t operator*() const noexcept {
    switch (State) {
    case PS_BeforeBegin:
    case PS_AtEnd:
      return "";
    case PS_InRootDir:
      return "/";
    case PS_InTrailingSep:
      return ".";
    case PS_InRootName:
    case PS_InFilenames:
      return RawEntry;
    }
    _LIBCPP_UNREACHABLE();
  }

  explicit operator bool() const noexcept {
    return State != PS_BeforeBegin && State != PS_AtEnd;
  }

  PathParser& operator++() noexcept {
    increment();
    return *this;
  }

  PathParser& operator--() noexcept {
    decrement();
    return *this;
  }

private:
  void makeState(ParserState NewState, PosPtr Start, PosPtr End) noexcept {
    State = NewState;
    RawEntry = string_view_t(Start, End - Start);
  }
  void makeState(ParserState NewState) noexcept {
    State = NewState;
    RawEntry = {};
  }

  PosPtr getAfterBack() const noexcept {
    return Path.data() + Path.size();
  }

  PosPtr getBeforeFront() const noexcept {
    return Path.data() - 1;
  }

  /// \brief Return a pointer to the first character after the currently
  ///   lexed element.
  PosPtr getNextTokenStartPos() const noexcept {
    switch (State) {
    case PS_BeforeBegin:
      return Path.data();
    case PS_InRootName:
    case PS_InRootDir:
    case PS_InFilenames:
      return &RawEntry.back() + 1;
    case PS_InTrailingSep:
    case PS_AtEnd:
      return getAfterBack();
    }
    _LIBCPP_UNREACHABLE();
  }

  /// \brief Return a pointer to the first character in the currently lexed
  ///   element.
  PosPtr getCurrentTokenStartPos() const noexcept {
    switch (State) {
    case PS_BeforeBegin:
    case PS_InRootName:
      return &Path.front();
    case PS_InRootDir:
    case PS_InFilenames:
    case PS_InTrailingSep:
      return &RawEntry.front();
    case PS_AtEnd:
      return &Path.back() + 1;
    }
    _LIBCPP_UNREACHABLE();
  }

  PosPtr consumeSeparator(PosPtr P, PosPtr End) const noexcept {
    if (P == End || *P != '/')
      return nullptr;
    const int Inc = P < End ? 1 : -1;
    P += Inc;
    while (P != End && *P == '/')
      P += Inc;
    return P;
  }

  PosPtr consumeName(PosPtr P, PosPtr End) const noexcept {
    if (P == End || *P == '/')
      return nullptr;
    const int Inc = P < End ? 1 : -1;
    P += Inc;
    while (P != End && *P != '/')
      P += Inc;
    return P;
  }
};

string_view_pair separate_filename(string_view_t const & s) {
    if (s == "." || s == ".." || s.empty()) return string_view_pair{s, ""};
    auto pos = s.find_last_of('.');
    if (pos == string_view_t::npos) return string_view_pair{s, string_view{}};
    return string_view_pair{s.substr(0, pos), s.substr(pos)};
}

string_view_t createView(PosPtr S, PosPtr E) noexcept {
  return {S, static_cast<size_t>(E - S) + 1};
}

}} // namespace parser

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_FILESYSTEM

using parser::string_view_t;
using parser::string_view_pair;
using parser::PathParser;
using parser::createView;

///////////////////////////////////////////////////////////////////////////////
//                            path definitions
///////////////////////////////////////////////////////////////////////////////

constexpr path::value_type path::preferred_separator;

path & path::replace_extension(path const & replacement)
{
    path p = extension();
    if (not p.empty()) {
      __pn_.erase(__pn_.size() - p.native().size());
    }
    if (!replacement.empty()) {
        if (replacement.native()[0] != '.') {
            __pn_ += ".";
        }
        __pn_.append(replacement.__pn_);
    }
    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// path.decompose

string_view_t path::__root_name() const
{
    auto PP = PathParser::CreateBegin(__pn_);
    if (PP.State == PathParser::PS_InRootName)
      return *PP;
    return {};
}

string_view_t path::__root_directory() const
{
    auto PP = PathParser::CreateBegin(__pn_);
    if (PP.State == PathParser::PS_InRootName)
      ++PP;
    if (PP.State == PathParser::PS_InRootDir)
      return *PP;
    return {};
}

string_view_t path::__root_path_raw() const
{
    auto PP = PathParser::CreateBegin(__pn_);
    if (PP.State == PathParser::PS_InRootName) {
      auto NextCh = PP.peek();
      if (NextCh && *NextCh == '/') {
        ++PP;
        return createView(__pn_.data(), &PP.RawEntry.back());
      }
      return PP.RawEntry;
    }
    if (PP.State == PathParser::PS_InRootDir)
      return *PP;
    return {};
}

string_view_t path::__relative_path() const
{
    auto PP = PathParser::CreateBegin(__pn_);
    while (PP.State <= PathParser::PS_InRootDir)
      ++PP;
    if (PP.State == PathParser::PS_AtEnd)
      return {};
    return createView(PP.RawEntry.data(), &__pn_.back());
}

string_view_t path::__parent_path() const
{
    if (empty())
      return {};
    auto PP = PathParser::CreateEnd(__pn_);
    --PP;
    if (PP.RawEntry.data() == __pn_.data())
      return {};
    --PP;
    return createView(__pn_.data(), &PP.RawEntry.back());
}

string_view_t path::__filename() const
{
    if (empty()) return {};
    return *(--PathParser::CreateEnd(__pn_));
}

string_view_t path::__stem() const
{
    return parser::separate_filename(__filename()).first;
}

string_view_t path::__extension() const
{
    return parser::separate_filename(__filename()).second;
}

////////////////////////////////////////////////////////////////////////////
// path.comparisons
int path::__compare(string_view_t __s) const {
    auto PP = PathParser::CreateBegin(__pn_);
    auto PP2 = PathParser::CreateBegin(__s);
    while (PP && PP2) {
        int res = (*PP).compare(*PP2);
        if (res != 0) return res;
        ++PP; ++PP2;
    }
    if (PP.State == PP2.State && PP.State == PathParser::PS_AtEnd)
        return 0;
    if (PP.State == PathParser::PS_AtEnd)
        return -1;
    return 1;
}

////////////////////////////////////////////////////////////////////////////
// path.nonmembers
size_t hash_value(const path& __p) noexcept {
  auto PP = PathParser::CreateBegin(__p.native());
  size_t hash_value = 0;
  std::hash<string_view> hasher;
  while (PP) {
    hash_value = __hash_combine(hash_value, hasher(*PP));
    ++PP;
  }
  return hash_value;
}

////////////////////////////////////////////////////////////////////////////
// path.itr
path::iterator path::begin() const
{
    auto PP = PathParser::CreateBegin(__pn_);
    iterator it;
    it.__path_ptr_ = this;
    it.__state_ = PP.State;
    it.__entry_ = PP.RawEntry;
    it.__stashed_elem_.__assign_view(*PP);
    return it;
}

path::iterator path::end() const
{
    iterator it{};
    it.__state_ = PathParser::PS_AtEnd;
    it.__path_ptr_ = this;
    return it;
}

path::iterator& path::iterator::__increment() {
  static_assert(__at_end == PathParser::PS_AtEnd, "");
  PathParser PP(__path_ptr_->native(), __entry_, __state_);
  ++PP;
  __state_ = PP.State;
  __entry_ = PP.RawEntry;
  __stashed_elem_.__assign_view(*PP);
  return *this;
}

path::iterator& path::iterator::__decrement() {
  PathParser PP(__path_ptr_->native(), __entry_, __state_);
  --PP;
  __state_ = PP.State;
  __entry_ = PP.RawEntry;
  __stashed_elem_.__assign_view(*PP);
  return *this;
}

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_FILESYSTEM
