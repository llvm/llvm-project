//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let currentStd = "all";
let currentStatus = "all";

function filterHeaders() {
  const searchEl = document.getElementById("header-search");
  const query = searchEl ? searchEl.value.toLowerCase().trim() : "";
  const cards = document.querySelectorAll(".header-card");
  let visible = 0;
  cards.forEach((card) => {
    const name = (card.dataset.name || "").toLowerCase();
    const stem = (card.dataset.stem || "").toLowerCase();
    const stds = (card.dataset.standard || "").split(" ");
    const status = card.dataset.status || "";
    const syms = (card.dataset.symbols || "").toLowerCase();
    const matchQ = !query || name.includes(query) || stem.includes(query) ||
                   syms.includes(query);
    const matchStd = currentStd === "all" || stds.includes(currentStd);
    const matchStatus = currentStatus === "all" || status === currentStatus;
    const show = matchQ && matchStd && matchStatus;
    card.style.display = show ? "" : "none";
    if (show)
      visible++;
  });
  const countEl = document.getElementById("filter-count");
  if (countEl)
    countEl.textContent = `Showing ${visible} of ${cards.length} headers`;
}

function setStandardFilter(val, btn) {
  currentStd = val;
  document.querySelectorAll(".filter-btn")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  filterHeaders();
}

function setStatusFilter(val, btn) {
  currentStatus = val;
  document.querySelectorAll(".status-btn")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  filterHeaders();
}

function sortHeaders() {
  const list = document.getElementById("headers-list");
  const sel = document.getElementById("sort-select");
  if (!list || !sel)
    return;
  const mode = sel.value;
  const cards = Array.from(list.querySelectorAll(".header-card"));
  cards.sort((a, b) => {
    const byName = (a.dataset.name || "").localeCompare(b.dataset.name || "");
    if (mode === "pct-desc")
      return parseFloat(b.dataset.pct) - parseFloat(a.dataset.pct) || byName;
    if (mode === "pct-asc")
      return parseFloat(a.dataset.pct) - parseFloat(b.dataset.pct) || byName;
    if (mode === "missing-desc")
      return (parseInt(b.dataset.missing, 10) -
                  parseInt(a.dataset.missing, 10) ||
              byName);
    return byName;
  });
  cards.forEach((c) => list.appendChild(c));
}

function toggleAllDetails(expand) {
  document.querySelectorAll(".header-card").forEach((d) => {
    if (d.style.display !== "none")
      d.open = expand;
  });
}
