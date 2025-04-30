#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee19  ########

ieee19: ieee19.$(OBJX)
	

ieee19.$(OBJX):  $(SRC)/ieee19.f90
	-$(RM) ieee19.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee19.f90 -o ieee19.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee19.$(OBJX) check.$(OBJX) $(LIBS) -o ieee19.$(EXESUFFIX)


ieee19.run: ieee19.$(OBJX)
	@echo ------------------------------------ executing test ieee19
	ieee19.$(EXESUFFIX)

verify: ;
build: ieee19.$(OBJX) ;
run: ieee19.run ;
