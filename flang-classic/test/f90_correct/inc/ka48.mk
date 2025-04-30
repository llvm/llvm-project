#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka48  ########


ka48: run
	

build:  $(SRC)/ka48.f
	-$(RM) ka48.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka48.f -o ka48.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka48.$(OBJX) check.$(OBJX) $(LIBS) -o ka48.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka48
	ka48.$(EXESUFFIX)

verify: ;

ka48.run: run

