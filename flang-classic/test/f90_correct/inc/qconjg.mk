#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test conjg  ########


qconjg: run


build:  $(SRC)/qconjg.f08
	-$(RM) qconjg.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qconjg.f08 -o qconjg.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qconjg.$(OBJX) check.$(OBJX) $(LIBS) -o qconjg.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qconjg
	qconjg.$(EXESUFFIX)

verify: ;

qconjg.run: run

