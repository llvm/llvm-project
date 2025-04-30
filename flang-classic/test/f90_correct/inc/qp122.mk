#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtosint  ########


qp122: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp122.f08 fcheck.$(OBJX)
	-$(RM) qp122.$(EXESUFFIX) qp122.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp122.f08 -o qp122.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp122.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp122.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp122
	qp122.$(EXESUFFIX)

verify: ;

qp122.run: run

