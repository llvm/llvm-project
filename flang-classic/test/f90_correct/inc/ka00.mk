#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka00  ########


ka00: run
	

build:  $(SRC)/ka00.f
	-$(RM) ka00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka00.f -o ka00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka00.$(OBJX) check.$(OBJX) $(LIBS) -o ka00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka00
	ka00.$(EXESUFFIX)

verify: ;

ka00.run: run

