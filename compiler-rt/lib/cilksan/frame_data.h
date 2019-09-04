// -*- C++ -*-
#ifndef __FRAME_DATA_H__
#define __FRAME_DATA_H__

#include "csan.h"
#include "disjointset.h"

enum EntryType_t : uint8_t { SPAWNER = 1, HELPER = 2, DETACHER = 3 };
enum FrameType_t : uint8_t { SHADOW_FRAME = 1, FULL_FRAME = 2, LOOP_FRAME = 3 };

typedef struct Entry_t {
  enum EntryType_t entry_type;
  enum FrameType_t frame_type;
  // const csi_id_t func_id;

  // fields that are for debugging purpose only
#if CILKSAN_DEBUG
  uint64_t frame_id;
  // uint32_t prev_helper; // index of the HELPER frame right above this entry
  //                     // initialized only for a HELPER frame
#endif
} Entry_t;

// Struct for keeping track of shadow frame
typedef struct FrameData_t {
  bool Sbag_used = false;
  bool Pbag_used = false;
  bool Iterbag_used = false;
  Entry_t frame_data;
  DisjointSet_t<SPBagInterface *> *Sbag = nullptr;
  DisjointSet_t<SPBagInterface *> *Pbag = nullptr;
  DisjointSet_t<SPBagInterface *> *Iterbag = nullptr;

  void set_sbag(DisjointSet_t<SPBagInterface *> *that) {
    if (Sbag)
      Sbag->dec_ref_count();

    Sbag = that;
    if (Sbag)
      Sbag->inc_ref_count();

    set_Sbag_used(false);
  }

  void set_pbag(DisjointSet_t<SPBagInterface *> *that) {
    if (Pbag)
      Pbag->dec_ref_count();

    Pbag = that;
    if (Pbag)
      Pbag->inc_ref_count();

    set_Pbag_used(false);
  }

  void set_iterbag(DisjointSet_t<SPBagInterface *> *that) {
    if (Iterbag)
      Iterbag->dec_ref_count();

    Iterbag = that;
    if (Iterbag)
      Iterbag->inc_ref_count();

    set_Iterbag_used(false);
  }

  void reset() {
    set_sbag(nullptr);
    set_pbag(nullptr);
    set_iterbag(nullptr);
  }

  FrameData_t() {}

  ~FrameData_t() {
    // update ref counts
    reset();
  }

  // Copy constructor and assignment operator ensure that reference
  // counts are properly maintained during resizing.
  FrameData_t(const FrameData_t &copy) : frame_data(copy.frame_data) {
    set_sbag(copy.Sbag);
    set_Sbag_used(copy.is_Sbag_used());
    set_pbag(copy.Pbag);
    set_Pbag_used(copy.is_Pbag_used());
    set_iterbag(copy.Iterbag);
    set_Iterbag_used(copy.is_Iterbag_used());
  }

  FrameData_t& operator=(const FrameData_t &copy) {
    frame_data = copy.frame_data;
    set_sbag(copy.Sbag);
    set_Sbag_used(copy.is_Sbag_used());
    set_pbag(copy.Pbag);
    set_Pbag_used(copy.is_Pbag_used());
    set_iterbag(copy.Iterbag);
    set_Iterbag_used(copy.is_Iterbag_used());
    return *this;
  }

  // remember to update this whenever new fields are added
  inline void init_new_function(DisjointSet_t<SPBagInterface *> *_sbag) {
    cilksan_assert(Pbag == NULL);
    set_sbag(_sbag);
  }

  bool is_Sbag_used() const { return Sbag_used; }
  bool is_Pbag_used() const { return Pbag_used; }
  bool is_Iterbag_used() const { return Iterbag_used; }
  void set_Sbag_used(bool v = true) { Sbag_used = v; }
  void set_Pbag_used(bool v = true) { Pbag_used = v; }
  void set_Iterbag_used(bool v = true) { Iterbag_used = v; }

  bool is_loop_frame() const {
    return (LOOP_FRAME == frame_data.frame_type);
  }

  void create_iterbag() {
    cilksan_assert(is_loop_frame());
    set_iterbag(new DisjointSet_t<SPBagInterface *>(
                    new SBag_t(Sbag->get_node()->get_func_id(),
                               *(Sbag->get_node()->get_call_stack()))));
  }
  bool inc_version() {
    cilksan_assert(nullptr != Iterbag);
    return Iterbag->get_node()->inc_version();
  }
  bool check_parallel_iter(const SPBagInterface *LCA, uint16_t version) const {
    if (!is_loop_frame())
      return false;
    cilksan_assert(nullptr != Iterbag);
    return ((LCA == Iterbag->get_node()) &&
            (version < Iterbag->get_node()->get_version()));
  }

  DisjointSet_t<SPBagInterface *> *getSbagForAccess() {
    if (!is_loop_frame()) {
      set_Sbag_used();
      return Sbag;
    }
    cilksan_assert(nullptr != Iterbag);
    set_Iterbag_used();
    return Iterbag;
  }

} FrameData_t;

#endif // __FRAME_DATA_H__
