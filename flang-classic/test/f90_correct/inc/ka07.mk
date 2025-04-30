#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka07  ########


ka07: run
	

build:  $(SRC)/ka07.f
	-$(RM) ka07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka07.f -o ka07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka07.$(OBJX) check.$(OBJX) $(LIBS) -o ka07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka07
	ka07.$(EXESUFFIX)

verify: ;

ka07.run: run

