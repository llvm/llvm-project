// -*- C++ -*-
#ifndef __FRAME_DATA_T__
#define __FRAME_DATA_T__

#include "csan.h"
#include "disjointset.h"

enum EntryType_t { SPAWNER = 1, HELPER = 2, DETACHER = 3 };
enum FrameType_t { SHADOW_FRAME = 1, FULL_FRAME = 2 };

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
  Entry_t frame_data;
  DisjointSet_t<SPBagInterface *> *Sbag;
  DisjointSet_t<SPBagInterface *> *Pbag;

  void set_sbag(DisjointSet_t<SPBagInterface *> *that) {
    if (Sbag)
      Sbag->dec_ref_count();

    Sbag = that;
    if (Sbag)
      Sbag->inc_ref_count();
  }

  void set_pbag(DisjointSet_t<SPBagInterface *> *that) {
    if (Pbag)
      Pbag->dec_ref_count();

    Pbag = that;
    if (Pbag)
      Pbag->inc_ref_count();
  }

  void reset() {
    set_sbag(NULL);
    set_pbag(NULL);
  }

  FrameData_t() :
    Sbag(NULL), Pbag(NULL) { }

  ~FrameData_t() {
    // update ref counts
    reset();
  }

  // Copy constructor and assignment operator ensure that reference
  // counts are properly maintained during resizing.
  FrameData_t(const FrameData_t &copy) :
    frame_data(copy.frame_data),
    Sbag(NULL), Pbag(NULL) {
    set_sbag(copy.Sbag);
    set_pbag(copy.Pbag);
  }

  FrameData_t& operator=(const FrameData_t &copy) {
    frame_data = copy.frame_data;
    set_sbag(copy.Sbag);
    set_pbag(copy.Pbag);
    return *this;
  }

  // remember to update this whenever new fields are added
  inline void init_new_function(DisjointSet_t<SPBagInterface *> *_sbag) {
    cilksan_assert(Pbag == NULL);
    set_sbag(_sbag);
  }

} FrameData_t;
#endif // __FRAME_DATA_T__
