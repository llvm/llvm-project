#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp19  ########


qp19: run


build:  $(SRC)/qp19.f08
	-$(RM) qp19.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp19.f08 -o qp19.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp19.$(OBJX) check.$(OBJX) $(LIBS) -o qp19.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp19
	qp19.$(EXESUFFIX)

verify: ;
