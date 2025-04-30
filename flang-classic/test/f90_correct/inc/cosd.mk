#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test cosd  ########


cosd: run


build:  $(SRC)/cosd.f08
	-$(RM) cosd.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/cosd.f08 -o cosd.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) cosd.$(OBJX) check.$(OBJX) $(LIBS) -o cosd.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test cosd
	cosd.$(EXESUFFIX)

verify: ;
