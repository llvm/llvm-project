#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka68  ########


ka68: run
	

build:  $(SRC)/ka68.f90
	-$(RM) ka68.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka68.f90 -o ka68.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka68.$(OBJX) check.$(OBJX) $(LIBS) -o ka68.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka68
	ka68.$(EXESUFFIX)

verify: ;

ka68.run: run

