#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka52  ########


ka52: run
	

build:  $(SRC)/ka52.f
	-$(RM) ka52.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka52.f -o ka52.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka52.$(OBJX) check.$(OBJX) $(LIBS) -o ka52.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka52
	ka52.$(EXESUFFIX)

verify: ;

ka52.run: run

