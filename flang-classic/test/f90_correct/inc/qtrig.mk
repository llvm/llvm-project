#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qsin qtan qcos  ########


qtrig: run
	

build:  $(SRC)/qtrig.f08
	-$(RM) qtrig.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qtrig.f08 -o qtrig.$(OBJX)
	-$(FC)  $(FFLAGS) $(LDFLAGS) qtrig.$(OBJX) check.$(OBJX) $(LIBS) -o qtrig.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qtrig 
	qtrig.$(EXESUFFIX)

verify: ;


