#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

$(TEST): run

build:  $(SRC)/$(TEST).f90
	-$(RM) $(TEST).$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(MKDIR) import_mod_from_user_dir1
	-$(CD) import_mod_from_user_dir1
	-$(CP) $(SRC)/import_mod_from_user_dir1_m2.f90 .
	-$(FC) -c import_mod_from_user_dir1_m2.f90
	-$(CD) ..
	-$(MKDIR) import_mod_from_user_dir2
	-$(CD) import_mod_from_user_dir2
	-$(CP) $(SRC)/import_mod_from_user_dir2* .
	-$(FC) -c import_mod_from_user_dir2_m2.f90
	-$(FC) -c import_mod_from_user_dir2_m1.f90
	-$(CD) ..
	-$(FC) -c -I./import_mod_from_user_dir1 -I./import_mod_from_user_dir2 $(SRC)/$(TEST).f90 -o $(TEST).$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) $(TEST).$(OBJX) $(LIBS) -o $(TEST).$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test $(TEST)
	$(TEST).$(EXESUFFIX)

verify: ;
