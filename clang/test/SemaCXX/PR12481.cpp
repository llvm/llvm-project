// RUN: %clang_cc1 -x c++ -fsyntax-only %s

class C1 { };
class C2 { };
template<class TrieData> struct BinaryTrie {
  ~BinaryTrie() {
    (void)(({
      static int x = 5;
    }
    ));
  }
};
class FooTable {
  BinaryTrie<C1> c1_trie_;
  BinaryTrie<C2> c2_trie_;
};
FooTable* foo = new FooTable;
