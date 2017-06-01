template < typename > struct vector {};

#include <map>
#include "Types.h"

struct TString {
   TString (char *);
};

struct TreeInfo {};

class DataInputHandler {
   void AddTree ();
   void SignalTreeInfo () {
      fInputTrees[(char*)""];
   }
   map <TString, vector <TreeInfo> >fInputTrees;
   map <string, bool> fExplicitTrainTest;
};
