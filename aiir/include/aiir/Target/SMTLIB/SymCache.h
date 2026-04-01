//===- SymCache.h - Declare Symbol Cache ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a Symbol Cache.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_SUPPORT_SYMCACHE_H
#define AIIR_SUPPORT_SYMCACHE_H

#include "aiir/IR/SymbolTable.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"

namespace aiir {

/// Base symbol cache class to allow for cache lookup through a pointer to some
/// abstract cache. A symbol cache stores lookup tables to make manipulating and
/// working with the IR more efficient.
class SymbolCacheBase {
public:
  virtual ~SymbolCacheBase();

  /// Defines 'op' as associated with the 'symbol' in the cache.
  virtual void addDefinition(aiir::Attribute symbol, aiir::Operation *op) = 0;

  /// Adds the symbol-defining 'op' to the cache.
  void addSymbol(aiir::SymbolOpInterface op) {
    addDefinition(op.getNameAttr(), op);
  }

  /// Populate the symbol cache with all symbol-defining operations within the
  /// 'top' operation.
  void addDefinitions(aiir::Operation *top);

  /// Lookup a definition for 'symbol' in the cache.
  virtual aiir::Operation *getDefinition(aiir::Attribute symbol) const = 0;

  /// Lookup a definition for 'symbol' in the cache.
  aiir::Operation *getDefinition(aiir::FlatSymbolRefAttr symbol) const {
    return getDefinition(symbol.getAttr());
  }

  /// Iterator support through a pointer to some abstract cache.
  /// The implementing cache must provide an iterator that carries values on the
  /// form of <aiir::Attribute, aiir::Operation*>.
  using CacheItem = std::pair<aiir::Attribute, aiir::Operation *>;
  struct CacheIteratorImpl {
    virtual ~CacheIteratorImpl() {}
    virtual void operator++() = 0;
    virtual CacheItem operator*() = 0;
    virtual bool operator==(CacheIteratorImpl *other) = 0;
  };

  struct Iterator
      : public llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                          CacheItem> {
    Iterator(std::unique_ptr<CacheIteratorImpl> &&impl)
        : impl(std::move(impl)) {}
    CacheItem operator*() const { return **impl; }
    using llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                     CacheItem>::operator++;
    bool operator==(const Iterator &other) const {
      return *impl == other.impl.get();
    }
    void operator++() { impl->operator++(); }

  private:
    std::unique_ptr<CacheIteratorImpl> impl;
  };
  virtual Iterator begin() = 0;
  virtual Iterator end() = 0;
};

/// Default symbol cache implementation; stores associations between names
/// (StringAttr's) to aiir::Operation's.
/// Adding/getting definitions from the symbol cache is not
/// thread safe. If this is required, synchronizing cache acccess should be
/// ensured by the caller.
class SymbolCache : public SymbolCacheBase {
public:
  /// In the building phase, add symbols.
  void addDefinition(aiir::Attribute key, aiir::Operation *op) override {
    symbolCache.try_emplace(key, op);
  }

  // Pull in getDefinition(aiir::FlatSymbolRefAttr symbol)
  using SymbolCacheBase::getDefinition;
  aiir::Operation *getDefinition(aiir::Attribute attr) const override {
    auto it = symbolCache.find(attr);
    if (it == symbolCache.end())
      return nullptr;
    return it->second;
  }

protected:
  /// This stores a lookup table from symbol attribute to the operation
  /// that defines it.
  llvm::DenseMap<aiir::Attribute, aiir::Operation *> symbolCache;

private:
  /// Iterator support: A simple mapping between decltype(symbolCache)::iterator
  /// to SymbolCacheBase::Iterator.
  using Iterator = decltype(symbolCache)::iterator;
  struct SymbolCacheIteratorImpl : public CacheIteratorImpl {
    SymbolCacheIteratorImpl(Iterator it) : it(it) {}
    CacheItem operator*() override { return {it->getFirst(), it->getSecond()}; }
    void operator++() override { it++; }
    bool operator==(CacheIteratorImpl *other) override {
      return it == static_cast<SymbolCacheIteratorImpl *>(other)->it;
    }
    Iterator it;
  };

public:
  SymbolCacheBase::Iterator begin() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<SymbolCacheIteratorImpl>(symbolCache.begin()));
  }
  SymbolCacheBase::Iterator end() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<SymbolCacheIteratorImpl>(symbolCache.end()));
  }
};

} // namespace aiir

#endif // AIIR_SUPPORT_SYMCACHE_H
