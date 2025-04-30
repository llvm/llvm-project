#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hg00  ########


hg00: run
	

build:  $(SRC)/hg00.f
	-$(RM) hg00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hg00.f -o hg00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hg00.$(OBJX) check.$(OBJX) $(LIBS) -o hg00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hg00
	hg00.$(EXESUFFIX)

verify: ;

hg00.run: run

