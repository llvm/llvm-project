#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka66  ########


ka66: run
	

build:  $(SRC)/ka66.f90
	-$(RM) ka66.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka66.f90 -o ka66.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka66.$(OBJX) check.$(OBJX) $(LIBS) -o ka66.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka66
	ka66.$(EXESUFFIX)

verify: ;

ka66.run: run

