
#
# Make rules to build compiler_rt in Apple B&I infrastructure
#

# set ProjSrcRoot appropriately
ProjSrcRoot := $(SRCROOT)
# set ProjObjRoot appropriately
ifdef OBJROOT
  ProjObjRoot := $(OBJROOT)
else
  ProjObjRoot := $(ProjSrcRoot)
endif

ifeq (,$(SDKROOT))
	INSTALL_TARGET = install-MacOSX
else
	INSTALL_TARGET = install-iOS
endif



# Log full compile lines in B&I logs and omit summary lines.
Verb :=
Summary := @true

# List of functions needed for each architecture.

# Copies any public headers to DSTROOT.
installhdrs:


# Copies source code to SRCROOT.
installsrc:
	cp -r . $(SRCROOT)


install:  $(INSTALL_TARGET)

# Copy results to DSTROOT.
install-MacOSX : $(SYMROOT)/libcompiler_rt.dylib \
                 $(SYMROOT)/libcompiler_rt-dyld.a 
	mkdir -p $(DSTROOT)/usr/local/lib/dyld
	cp $(SYMROOT)/libcompiler_rt-dyld.a  \
				    $(DSTROOT)/usr/local/lib/dyld/libcompiler_rt.a
	mkdir -p $(DSTROOT)/usr/lib/system
	$(call GetCNAVar,STRIP,Platform.darwin_bni,Release,) -S $(SYMROOT)/libcompiler_rt.dylib \
	    -o $(DSTROOT)/usr/lib/system/libcompiler_rt.dylib
	cd $(DSTROOT)/usr/lib/system; \
	    ln -s libcompiler_rt.dylib libcompiler_rt_profile.dylib; \
	    ln -s libcompiler_rt.dylib libcompiler_rt_debug.dylib

# Rule to make each dylib slice
$(OBJROOT)/libcompiler_rt-%.dylib : $(OBJROOT)/darwin_bni/Release/%/libcompiler_rt.a
	echo "const char vers[] = \"@(#) $(RC_ProjectName)-$(RC_ProjectSourceVersion)\"; " > $(OBJROOT)/version.c
	$(call GetCNAVar,CC,Platform.darwin_bni,Release,$*) \
	   $(OBJROOT)/version.c -arch $* -dynamiclib \
	   -install_name /usr/lib/system/libcompiler_rt.dylib \
	   -compatibility_version 1 -current_version $(RC_ProjectSourceVersion) \
	   -nodefaultlibs -lSystem -umbrella System -dead_strip \
	   $(DYLIB_FLAGS) -Wl,-force_load,$^ -o $@ 

# Rule to make fat dylib
$(SYMROOT)/libcompiler_rt.dylib: $(foreach arch,$(filter-out armv4t,$(RC_ARCHS)), \
                                        $(OBJROOT)/libcompiler_rt-$(arch).dylib)
	$(call GetCNAVar,LIPO,Platform.darwin_bni,Release,) -create $^ -o  $@




# Copy results to DSTROOT.
install-iOS: $(SYMROOT)/libcompiler_rt-static.a \
             $(SYMROOT)/libcompiler_rt-dyld.a \
             $(SYMROOT)/libcompiler_rt.dylib
	mkdir -p $(DSTROOT)/usr/local/lib
	cp $(SYMROOT)/libcompiler_rt-static.a  \
				    $(DSTROOT)/usr/local/lib/libcompiler_rt-static.a
	mkdir -p $(DSTROOT)/usr/local/lib/dyld
	cp $(SYMROOT)/libcompiler_rt-dyld.a  \
				    $(DSTROOT)/usr/local/lib/dyld/libcompiler_rt.a
	mkdir -p $(DSTROOT)/usr/lib/system
	$(call GetCNAVar,STRIP,Platform.darwin_bni,Release,) -S $(SYMROOT)/libcompiler_rt.dylib \
	    -o $(DSTROOT)/usr/lib/system/libcompiler_rt.dylib

	
# Rule to make fat archive
$(SYMROOT)/libcompiler_rt-static.a : $(foreach arch,$(RC_ARCHS), \
                         $(OBJROOT)/darwin_bni/Static/$(arch)/libcompiler_rt.a)
	$(call GetCNAVar,LIPO,Platform.darwin_bni,Release,) -create $^ -o  $@

# rule to make each archive slice for dyld (which removes a few archive members)
$(OBJROOT)/libcompiler_rt-dyld-%.a : $(OBJROOT)/darwin_bni/Release/%/libcompiler_rt.a
	cp $^ $@
	DEL_LIST=`$(AR)  -t $@ | egrep 'apple_versioning|gcc_personality_v0|eprintf' | xargs echo` ; \
	if [ -n "$${DEL_LIST}" ] ; \
	then  \
		$(call GetCNAVar,AR,Platform.darwin_bni,Release,) -d $@ $${DEL_LIST}; \
		$(call GetCNAVar,RANLIB,Platform.darwin_bni,Release,) $@ ; \
	fi

# rule to make make archive for dyld
$(SYMROOT)/libcompiler_rt-dyld.a : $(foreach arch,$(RC_ARCHS), \
                         $(OBJROOT)/libcompiler_rt-dyld-$(arch).a)
	$(call GetCNAVar,LIPO,Platform.darwin_bni,Release,) -create $^ -o  $@

