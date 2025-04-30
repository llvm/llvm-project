#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do11: do11.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do11.$(OBJX): $(SRC)/do11.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do11.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do11.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do11
run: ;
