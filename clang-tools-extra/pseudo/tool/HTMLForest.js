// The global map of forest node index => NodeView.
views = [];
// NodeView is a visible forest node.
// It has an entry in the navigation tree, and a span in the code itself.
// Each NodeView is associated with a forest node, but not all nodes have views:
// - nodes not reachable though current ambiguity selection
// - trivial "wrapping" sequence nodes are abbreviated away
class NodeView {
  // Builds a node representing forest[index], or its target if it is a wrapper.
  // Registers the node in the global map.
  static make(index, parent, abbrev) {
    var node = forest[index];
    if (node.kind == 'sequence' && node.children.length == 1 &&
        forest[node.children[0]].kind != 'ambiguous') {
      abbrev ||= [];
      abbrev.push(index);
      return NodeView.make(node.children[0], parent, abbrev);
    }
    return views[index] = new NodeView(index, parent, node, abbrev);
  }

  constructor(index, parent, node, abbrev) {
    this.abbrev = abbrev || [];
    this.parent = parent;
    this.children =
        (node.kind == 'ambiguous' ? [ node.selected ] : node.children || [])
            .map((c) => NodeView.make(c, this));
    this.index = index;
    this.node = node;
    views[index] = this;

    this.span = this.buildSpan();
    this.tree = this.buildTree();
  }

  // Replaces the token sequence in #code with a <span class=node>.
  buildSpan() {
    var elt = document.createElement('span');
    elt.dataset['index'] = this.index;
    elt.classList.add("node");
    elt.classList.add("selectable-node");
    elt.classList.add(this.node.kind);

    var begin = null, end = null;
    if (this.children.length != 0) {
      begin = this.children[0].span;
      end = this.children[this.children.length - 1].span.nextSibling;
    } else if (this.node.kind == 'terminal') {
      begin = document.getElementById(this.node.token);
      end = begin.nextSibling;
    } else if (this.node.kind == 'opaque') {
      begin = document.getElementById(this.node.firstToken);
      end = (this.node.lastToken == null)
                ? begin
                : document.getElementById(this.node.lastToken).nextSibling;
    }
    var parent = begin.parentNode;
    splice(begin, end, elt);
    parent.insertBefore(elt, end);
    return elt;
  }

  // Returns a (detached) <li class=tree-node> suitable for use in #tree.
  buildTree() {
    var elt = document.createElement('li');
    elt.dataset['index'] = this.index;
    elt.classList.add('tree-node');
    elt.classList.add('selectable-node');
    elt.classList.add(this.node.kind);
    var header = document.createElement('header');
    elt.appendChild(header);

    if (this.abbrev.length > 0) {
      var abbrev = document.createElement('span');
      abbrev.classList.add('abbrev');
      abbrev.innerText = forest[this.abbrev[0]].symbol;
      header.appendChild(abbrev);
    }
    var name = document.createElement('span');
    name.classList.add('name');
    name.innerText = this.node.symbol;
    header.appendChild(name);

    if (this.children.length != 0) {
      var sublist = document.createElement('ul');
      this.children.forEach((c) => sublist.appendChild(c.tree));
      elt.appendChild(sublist);
    }
    return elt;
  }

  // Make this view visible on the screen by scrolling if needed.
  scrollVisible() {
    scrollIntoViewV(document.getElementById('tree'), this.tree.firstChild);
    scrollIntoViewV(document.getElementById('code'), this.span);
  }

  // Fill #info with details of this node.
  renderInfo() {
    document.getElementById('info').classList = this.node.kind;
    document.getElementById('i_symbol').innerText = this.node.symbol;
    document.getElementById('i_kind').innerText = this.node.kind;

    // For sequence nodes, add LHS := RHS rule.
    // If this node abbreviates trivial sequences, we want those rules too.
    var rules = document.getElementById('i_rules');
    rules.textContent = '';
    function addRule(i) {
      var ruleText = forest[i].rule;
      if (ruleText == null)
        return;
      var rule = document.createElement('div');
      rule.classList.add('rule');
      rule.innerText = ruleText;
      rules.insertBefore(rule, rules.firstChild);
    }
    this.abbrev.forEach(addRule);
    addRule(this.index);

    // For ambiguous nodes, show a selectable list of alternatives.
    var alternatives = document.getElementById('i_alternatives');
    alternatives.textContent = '';
    var that = this;
    function addAlternative(i) {
      var altNode = forest[i];
      var text = altNode.rule || altNode.kind;
      var alt = document.createElement('div');
      alt.classList.add('alternative');
      alt.innerText = text;
      alt.dataset['index'] = i;
      alt.dataset['parent'] = that.index;
      if (i == that.node.selected)
        alt.classList.add('selected');
      alternatives.appendChild(alt);
    }
    if (this.node.kind == 'ambiguous')
      this.node.children.forEach(addAlternative);

    // Show the stack of ancestor nodes.
    // The part of each rule that leads to the current node is bolded.
    var ancestors = document.getElementById('i_ancestors');
    ancestors.textContent = '';
    var child = this;
    for (var view = this.parent; view != null;
         child = view, view = view.parent) {
      var indexInParent = view.children.indexOf(child);

      var ctx = document.createElement('div');
      ctx.classList.add('ancestors');
      ctx.classList.add('selectable-node');
      ctx.classList.add(view.node.kind);
      if (view.node.rule) {
        // Rule syntax is LHS := RHS1 [annotation] RHS2.
        // We walk through the chunks and bold the one at parentInIndex.
        var chunkCount = 0;
        ctx.innerHTML = view.node.rule.replaceAll(/[^ ]+/g, function(match) {
          if (!(match.startsWith('[') && match.endsWith(']')) /*annotations*/
              && chunkCount++ == indexInParent + 2 /*skip LHS :=*/)
            return '<b>' + match + '</b>';
          return match;
        });
      } else /*ambiguous*/ {
        ctx.innerHTML = '<b>' + view.node.symbol + '</b>';
      }
      ctx.dataset['index'] = view.index;
      if (view.abbrev.length > 0) {
        var abbrev = document.createElement('span');
        abbrev.classList.add('abbrev');
        abbrev.innerText = forest[view.abbrev[0]].symbol;
        ctx.insertBefore(abbrev, ctx.firstChild);
      }

      ctx.dataset['index'] = view.index;
      ancestors.appendChild(ctx, ancestors.firstChild);
    }
  }

  remove() {
    this.children.forEach((c) => c.remove());
    splice(this.span.firstChild, null, this.span.parentNode,
           this.span.nextSibling);
    detach(this.span);
    delete views[this.index];
  }
};

var selection = null;
function selectView(view) {
  var old = selection;
  selection = view;
  if (view == old)
    return;

  if (old) {
    old.tree.classList.remove('selected');
    old.span.classList.remove('selected');
  }
  document.getElementById('info').hidden = (view == null);
  if (!view)
    return;
  view.tree.classList.add('selected');
  view.span.classList.add('selected');
  view.renderInfo();
  view.scrollVisible();
}

// To highlight nodes on hover, we create dynamic CSS rules of the form
//   .selectable-node[data-index="42"] { background-color: blue; }
// This avoids needing to find all the related nodes and update their classes.
var highlightSheet = new CSSStyleSheet();
document.adoptedStyleSheets.push(highlightSheet);
function highlightView(view) {
  var text = '';
  for (const color of ['#6af', '#bbb', '#ddd', '#eee']) {
    if (view == null)
      break;
    text += '.selectable-node[data-index="' + view.index + '"] '
    text += '{ background-color: ' + color + '; }\n';
    view = view.parent;
  }
  highlightSheet.replace(text);
}

// Select which branch of an ambiguous node is taken.
function chooseAlternative(parent, index) {
  var parentView = views[parent];
  parentView.node.selected = index;
  var oldChild = parentView.children[0];
  oldChild.remove();
  var newChild = NodeView.make(index, parentView);
  parentView.children[0] = newChild;
  parentView.tree.lastChild.replaceChild(newChild.tree, oldChild.tree);

  highlightView(null);
  // Force redraw of the info box.
  selectView(null);
  selectView(parentView);
}

// Attach event listeners and build content once the document is ready.
document.addEventListener("DOMContentLoaded", function() {
  var code = document.getElementById('code');
  var tree = document.getElementById('tree');
  var ancestors = document.getElementById('i_ancestors');
  var alternatives = document.getElementById('i_alternatives');

  [code, tree, ancestors].forEach(function(container) {
    container.addEventListener('click', function(e) {
      var nodeElt = e.target.closest('.selectable-node');
      selectView(nodeElt && views[Number(nodeElt.dataset['index'])]);
    });
    container.addEventListener('mousemove', function(e) {
      var nodeElt = e.target.closest('.selectable-node');
      highlightView(nodeElt && views[Number(nodeElt.dataset['index'])]);
    });
  });

  alternatives.addEventListener('click', function(e) {
    var altElt = e.target.closest('.alternative');
    if (altElt)
      chooseAlternative(Number(altElt.dataset['parent']),
                        Number(altElt.dataset['index']));
  });

  // The HTML provides #code content in a hidden DOM element, move it.
  var hiddenCode = document.getElementById('hidden-code');
  splice(hiddenCode.firstChild, hiddenCode.lastChild, code);
  detach(hiddenCode);

  // Build the tree of NodeViews and attach to #tree.
  tree.firstChild.appendChild(NodeView.make(0).tree);
});

// Helper DOM functions //

// Moves the sibling range [first, until) into newParent.
function splice(first, until, newParent, before) {
  for (var next = first; next != until;) {
    var elt = next;
    next = next.nextSibling;
    newParent.insertBefore(elt, before);
  }
}
function detach(node) { node.parentNode.removeChild(node); }
// Like scrollIntoView, but vertical only!
function scrollIntoViewV(container, elt) {
  if (container.scrollTop > elt.offsetTop + elt.offsetHeight ||
      container.scrollTop + container.clientHeight < elt.offsetTop)
    container.scrollTo({top : elt.offsetTop, behavior : 'smooth'});
}
