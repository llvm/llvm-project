#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test power02  ########


power02: run


build:  $(SRC)/power02.f08
	-$(RM) power02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/power02.f08 -o power02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) power02.$(OBJX) check.$(OBJX) $(LIBS) -o power02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test power02
	power02.$(EXESUFFIX)

verify: ;
